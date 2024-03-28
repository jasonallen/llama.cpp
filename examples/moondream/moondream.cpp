#include "vision_encoder.h"
#include "common.h"
#include "llama.h"
#include "moondream.h"
#include "base64.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>


// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

struct clip_image_grid_shape {
    int first;
    int second;
};

/**
 * Selects the best resolution from a list of possible resolutions based on the original size.
 *
 * @param original_size The original size of the image in the format (width, height).
 * @param possible_resolutions A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].
 * @return The best fit resolution in the format (width, height).
 */
static std::pair<int, int> select_best_resolution(const std::pair<int, int>& original_size, const std::vector<std::pair<int, int>>& possible_resolutions) {
    int original_width  = original_size.first;
    int original_height = original_size.second;

    std::pair<int, int> best_fit;
    int max_effective_resolution = 0;
    int min_wasted_resolution = std::numeric_limits<int>::max();

    for (const auto& resolution : possible_resolutions) {
        int width = resolution.first;
        int height = resolution.second;
        float scale = std::min(static_cast<float>(width) / original_width, static_cast<float>(height) / original_height);
        int downscaled_width  = static_cast<int>(original_width * scale);
        int downscaled_height = static_cast<int>(original_height * scale);
        int effective_resolution = std::min(downscaled_width * downscaled_height, original_width * original_height);
        int wasted_resolution = (width * height) - effective_resolution;
        // fprintf(stderr, "resolution: %d %d, scale: %f, downscaled: %d %d, effective: %d, wasted: %d\n", width, height, scale, downscaled_width, downscaled_height, effective_resolution, wasted_resolution);
        if (effective_resolution > max_effective_resolution || (effective_resolution == max_effective_resolution && wasted_resolution < min_wasted_resolution)) {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
            best_fit = resolution;
        }
    }

    return best_fit;
}

/**
 * @brief Get the anyres image grid shape object
 *
 * @param image_size
 * @param grid_pinpoints
 * @param image_patch_size
 * @return <int, int>
 */
static struct clip_image_grid_shape get_anyres_image_grid_shape(const std::pair<int, int> & image_size, const std::vector<std::pair<int, int>> & grid_pinpoints, int image_patch_size) {
    /**
        Conversion from gguf flat array to vector:
        std::vector<std::pair<int, int>> possible_resolutions;
        for (int i = 0; i < 32 && params.image_grid_pinpoints[i] != 0; i+=2) {
            possible_resolutions.push_back({params.image_grid_pinpoints[i], params.image_grid_pinpoints[i+1]});
        }
     */
    auto best_resolution = select_best_resolution(image_size, grid_pinpoints);
    return {best_resolution.first / image_patch_size, best_resolution.second / image_patch_size};
}

// Take the image segments in a grid configuration and return the embeddings and the number of embeddings into preallocated memory (image_embd_out)
static bool clip_moondream_handle_patches(clip_ctx * ctx_clip, std::vector<float *> & image_embd_v, struct clip_image_grid_shape grid_shape, float * image_embd_out, int * n_img_pos_out) {
    struct {
        struct ggml_tensor * newline;
        struct ggml_context * ctx;
    } model;

    const int32_t image_size = clip_image_size(ctx_clip);
    const int32_t patch_size = clip_patch_size(ctx_clip);

    int32_t num_patches_per_side = image_size / patch_size; // 336 / 14 = 24 - used for embedding-patching boxes (24*24 = 576 patches)

    int num_patches_width  = grid_shape.first;  // grid 1-4
    int num_patches_height = grid_shape.second; // grid 1-4

    const size_t num_images = num_patches_width * num_patches_height + 1;

    // TODO: size calculation is not calculated - it's only tens of MB
    size_t ctx_size = 0;

    {
        ctx_size += clip_embd_nbytes(ctx_clip) * num_images * 8; // image_features
        ctx_size += 1024*1024 * ggml_type_size(GGML_TYPE_F32);
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // Python reference code for full unpad:
    /*
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = unpad_image(image_feature, image_sizes[image_idx])
        image_feature = torch.cat((
            image_feature,
            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)
        ), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    */
    // We now have two options: unpad or no unpad. Unpad removes tokens for faster llm eval.
    // In terms of result quality it appears to make no difference, so we'll start with the easier approach given 5D tensors are not supported in ggml yet.
    // Without unpad we have to split the sub-image embeddings into patches of 24 features each and permute them.
    // Once all images are processed to prepended the base_image_features without any changes.

    // Pytorch reference simplified, modified for ggml compatibility - confirmed identical output in python (for a 2x2 grid image (676x676 scaling))
    /*
        image_feature = image_feature.view(2, 2, 24, 24, 4096)
        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
        image_feature = image_feature.view(2, 24, 2, 24, 4096)
        image_feature = image_feature.flatten(0, 3)

        // Reshape to 4D tensor by merging the last two dimensions
        image_feature = image_feature.view(2, 2, 24, 24*4096)
        image_feature = image_feature.permute(0, 2, 1, 3).contiguous()
        image_feature = image_feature.view(-1, 4096)
    */

    model.ctx = ggml_init(params);

    ggml_tensor * newline_tmp = clip_get_newline_tensor(ctx_clip);
    model.newline = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, newline_tmp->ne[0]);
    if (newline_tmp->backend != GGML_BACKEND_TYPE_CPU) {
        if (newline_tmp->buffer == NULL) {
            printf("newline_tmp tensor buffer is NULL\n");
        }
        ggml_backend_tensor_get(newline_tmp, model.newline->data, 0, ggml_nbytes(newline_tmp));
    } else {
        model.newline->data = newline_tmp->data;
        if (model.newline->data == NULL) {
            printf("newline_tmp tensor data is NULL\n");
        }
    }

    struct ggml_tensor * image_features = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, clip_n_mmproj_embd(ctx_clip), clip_n_patches(ctx_clip), num_images - 1); // example: 4096 x 576 x 4
    // ggml_tensor_printf(image_features,"image_features",__LINE__,false,false);
    // fill it with the image embeddings, ignoring the base
    for (size_t i = 1; i < num_images; i++) {
        size_t offset = (i-1) * clip_embd_nbytes(ctx_clip);
        memcpy((uint8_t *)(image_features->data) + offset, image_embd_v[i], clip_embd_nbytes(ctx_clip));
    }

    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);
    size_t size_ele = ggml_type_size(GGML_TYPE_F32);

    struct ggml_tensor *image_features_patchview = ggml_view_4d(model.ctx, image_features,
                                                                num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
                                                                num_patches_per_side,
                                                                num_patches_width,
                                                                num_patches_height,
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip),
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side,
                                                                size_ele * num_patches_per_side * clip_n_mmproj_embd(ctx_clip) * num_patches_per_side * num_patches_width, 0);
    // ggml_tensor_printf(image_features_patchview,"image_features_patchview",__LINE__,false,false);
    struct ggml_tensor *permuted_cont = ggml_cont(model.ctx, ggml_permute(model.ctx, image_features_patchview, 0, 2, 1, 3));
    /**
     At the end of each row we have to add the row_end embeddings, which are the same as the newline embeddings
         image_feature = torch.cat((
        image_feature,
        self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    ), dim=-1)
     *
     */

    // ggml_tensor_printf(permuted_cont,"permuted_cont",__LINE__,false,false);
    struct ggml_tensor *flatten = ggml_view_2d(model.ctx, permuted_cont, clip_n_mmproj_embd(ctx_clip), num_patches_height * num_patches_width * num_patches_per_side * num_patches_per_side,  size_ele * clip_n_mmproj_embd(ctx_clip), 0);
    // ggml_tensor_printf(flatten,"flatten",__LINE__,false,false);
    ggml_build_forward_expand(gf, flatten);
    ggml_graph_compute_with_ctx(model.ctx, gf, 1);
    struct ggml_tensor* result = gf->nodes[gf->n_nodes - 1];

    memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as global context
    // append without newline tokens (default behavior in llava_arch when not using unpad ):
    memcpy(image_embd_out + clip_n_patches(ctx_clip) * clip_n_mmproj_embd(ctx_clip), (float*)result->data, clip_embd_nbytes(ctx_clip) * (num_images-1)); // grid patches
    *n_img_pos_out = static_cast<int>(result->ne[1]+clip_n_patches(ctx_clip));

    // Debug: Test single segments
    // Current findings: sending base image, sending a segment embedding all works similar to python
    // However, permuted embeddings do not work yet (stride issue?)
    // memcpy(image_embd_out, image_embd_v[0], clip_embd_nbytes(ctx_clip)); // main image as context
    // memcpy(image_embd_out, (float*)prepared_cont->data, clip_embd_nbytes(ctx_clip)); // main image as context
    // *n_img_pos_out=576;

    ggml_free(model.ctx);
    return true;
}


static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_pos) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(ctx_clip, img, img_res_v)) {
        fprintf(stderr, "%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    const int64_t t_img_enc_start_us = ggml_time_us();

    const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);

    if (strcmp(mm_patch_merge_type, "spatial_unpad") != 0) {
        // flat / default llava-1.5 type embedding
        *n_img_pos = clip_n_patches(ctx_clip);
        bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[0], image_embd); // image_embd shape is 576 x 4096
        delete[] img_res_v.data;
        if (!encoded) {
            fprintf(stderr, "Unable to encode image\n");

            return false;
        }
    } else {
        // spatial_unpad llava-1.6 type embedding
        // TODO: CLIP needs batching support - in HF the llm projection is separate after encoding, which might be a solution to quickly get batching working
        std::vector<float *> image_embd_v;
        image_embd_v.resize(img_res_v.size);
        for (size_t i = 0; i < img_res_v.size; i++) {
            image_embd_v[i] = (float *)malloc(clip_embd_nbytes(ctx_clip)); // 576 patches * 4096 embeddings * 4 bytes = 9437184
            const bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[i], image_embd_v[i]); // image data is in 3x336x336 format and will be converted to 336x336x3 inside
            if (!encoded) {
                fprintf(stderr, "Unable to encode image - spatial_unpad - subimage %d of %d\n", (int) i+1, (int) img_res_v.size);
                return false;
            }
        }
        const int64_t t_img_enc_batch_us = ggml_time_us();
        printf("%s: %d segments encoded in %8.2f ms\n", __func__, (int)img_res_v.size, (t_img_enc_batch_us - t_img_enc_start_us) / 1000.0);

        const int32_t * image_grid = clip_image_grid(ctx_clip);

        std::vector<std::pair<int, int>> grid_pinpoints;
        for (int i = 0; i < 32 && image_grid[i] != 0; i += 2) {
            grid_pinpoints.push_back({image_grid[i], image_grid[i+1]});
        }

        // free all img_res_v - not needed anymore
        delete[] img_res_v.data;
        img_res_v.size = 0;
        img_res_v.data = nullptr;

        const int32_t image_size = clip_image_size(ctx_clip);

        struct clip_image_grid_shape grid_shape = get_anyres_image_grid_shape({img->nx,img->ny}, grid_pinpoints, image_size);

        int n_img_pos_out;
        clip_moondream_handle_patches(ctx_clip, image_embd_v, grid_shape, image_embd, &n_img_pos_out);
        *n_img_pos = n_img_pos_out;

        for (size_t i = 0; i < image_embd_v.size(); i++) {
            free(image_embd_v[i]);
        }
        image_embd_v.clear();

        // debug image/segment/normalization content:
        // clip_image_u8 * tmp = clip_image_u8_init();
        // clip_image_convert_f32_to_u8(*image_feature, *tmp);
        // clip_image_save_to_bmp(*tmp, "image_feature.bmp");
    }

    printf("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    printf("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

bool moondream_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip) {
        // make sure that the correct mmproj was used, i.e., compare apples to apples
    int n_llama_embd = llama_n_embd(llama_get_model(ctx_llama));
    auto n_image_embd = clip_n_mmproj_embd(ctx_clip);
    if (n_image_embd != n_llama_embd) {
        printf("%s: embedding dim of the multimodal projector (%d) is not equal to that of LLaMA (%d). Make sure that you use the correct mmproj file.\n", __func__, n_image_embd, n_llama_embd);
        return false;
    }
    return true;
}

bool moondream_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out) {
    float * image_embd = (float *)malloc(clip_embd_nbytes(ctx_clip)*6); // TODO: base on gridsize/llava model
    if (!image_embd) {
        fprintf(stderr, "Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos)) {
        fprintf(stderr, "%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

bool moondream_eval_image_embed(llama_context * ctx_llama, const struct moondream_image_embed * image_embed, int n_batch, int * n_past) {
    int n_embd  = llama_n_embd(llama_get_model(ctx_llama));

    for (int i = 0; i < image_embed->n_image_pos; i += n_batch) {
        int n_eval = image_embed->n_image_pos - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        llama_batch batch = {int32_t(n_eval), nullptr, (image_embed->embed+i*n_embd), nullptr, nullptr, nullptr, nullptr, *n_past, 1, 0, };
        if (llama_decode(ctx_llama, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

struct moondream_image_embed * moondream_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length) {
    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = moondream_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        fprintf(stderr, "%s: coulnd't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (moondream_image_embed*)malloc(sizeof(moondream_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

static bool load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        fprintf(stderr, "%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        die_fmt("read error: %s", strerror(errno));
    }
    if (ret != (size_t) fileSize) {
        die("unexpectedly reached end of file");
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}

struct moondream_image_embed * moondream_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path) {
    unsigned char* image_bytes;
    long image_bytes_length;
    auto loaded = load_file_to_bytes(image_path, &image_bytes, &image_bytes_length);
    if (!loaded) {
        fprintf(stderr, "%s: failed to load %s\n", __func__, image_path);
        return NULL;
    }

    moondream_image_embed *embed = moondream_image_embed_make_with_bytes(ctx_clip, n_threads, image_bytes, image_bytes_length);
    free(image_bytes);

    return embed;
}


void moondream_image_embed_free(struct moondream_image_embed * embed) {
    free(embed->embed);
    free(embed);
}

struct llama_context * new_llama_context(moondream_context* md_ctx) {
    struct gpt_params *gpt_params = md_ctx->gpt_params;
    llama_context_params ctx_params = llama_context_params_from_gpt_params(*gpt_params);
    ctx_params.n_ctx = gpt_params->n_ctx < 2048 ? 2048 : gpt_params->n_ctx; // we need a longer context size to process image embeddings

    llama_context * ctx_llama = llama_new_context_with_model(md_ctx->model, ctx_params);

    if (ctx_llama == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }
    return ctx_llama;
}

struct moondream_context * moondream_init(struct gpt_params &gpt_params) {
    auto ctx_clip = clip_model_load(gpt_params.mmproj.c_str(), /*verbosity=*/ 1);
    llama_backend_init();
    llama_numa_init(gpt_params.numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(gpt_params);
    llama_model * model = llama_load_model_from_file(gpt_params.model.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return NULL;
    }

    auto ctx_md = (struct moondream_context *)malloc(sizeof(moondream_context));
    ctx_md->gpt_params = &gpt_params;
    ctx_md->ctx_clip = ctx_clip;
    ctx_md->model = model;
    return ctx_md;
}

static const char* IMG_BASE64_TAG_BEGIN = "<img src=\"data:image/jpeg;base64,";
static const char* IMG_BASE64_TAG_END = "\">";

static void find_image_tag_in_prompt(const std::string& prompt, size_t& begin_out, size_t& end_out) {
    begin_out = prompt.find(IMG_BASE64_TAG_BEGIN);
    end_out = prompt.find(IMG_BASE64_TAG_END, (begin_out == std::string::npos) ? 0UL : begin_out);
}


 bool prompt_contains_image(std::string prompt) {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    return (begin != std::string::npos);
}


// replaces the base64 image tag in the prompt with `replacement`
static moondream_image_embed * moondream_image_embed_make_with_prompt_base64(struct clip_ctx * ctx_clip, int n_threads, const std::string& prompt) {
    size_t img_base64_str_start, img_base64_str_end;
    find_image_tag_in_prompt(prompt, img_base64_str_start, img_base64_str_end);
    if (img_base64_str_start == std::string::npos || img_base64_str_end == std::string::npos) {
        fprintf(stderr, "%s: invalid base64 image tag. must be %s<base64 byte string>%s\n", __func__, IMG_BASE64_TAG_BEGIN, IMG_BASE64_TAG_END);
        return NULL;
    }

    auto base64_bytes_start = img_base64_str_start + strlen(IMG_BASE64_TAG_BEGIN);
    auto base64_bytes_count = img_base64_str_end - base64_bytes_start;
    auto base64_str = prompt.substr(base64_bytes_start, base64_bytes_count );

    auto required_bytes = base64::required_encode_size(base64_str.size());
    auto img_bytes = std::vector<unsigned char>(required_bytes);
    base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

    auto embed = moondream_image_embed_make_with_bytes(ctx_clip, n_threads, img_bytes.data(), img_bytes.size());
    if (!embed) {
        fprintf(stderr, "%s: could not load image from base64 string.\n", __func__);
        return NULL;
    }

    return embed;
}


static bool endsWith(const std::string& fullString, const std::string& ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}


static std::string remove_image_from_prompt(const std::string& prompt, const char * replacement = "") {
    size_t begin, end;
    find_image_tag_in_prompt(prompt, begin, end);
    if (begin == std::string::npos || end == std::string::npos) {
        return prompt;
    }
    auto pre = prompt.substr(0, begin);
    auto post = prompt.substr(end + strlen(IMG_BASE64_TAG_END));
    return pre + replacement + post;
}


void load_image(moondream_context * ctx_md, std::string& image_path, std::string& prompt, int32_t n_threads, struct image_embed_result &result) {
    result.embed = NULL;
    result.prompt = prompt;

    if (prompt_contains_image(prompt)) {
        if (!image_path.empty()) {
            fprintf(stderr, "using base64 encoded image instead of command line image path\n");
        }
        result.embed = moondream_image_embed_make_with_prompt_base64(ctx_md->ctx_clip, n_threads, prompt);
        if (!result.embed) {
            fprintf(stderr, "%s: can't load image from prompt\n", __func__);
            return;
        }
        result.prompt = remove_image_from_prompt(prompt);
    } else {
        result.embed = moondream_image_embed_make_with_filename(ctx_md->ctx_clip, n_threads, image_path.c_str());
        if (!result.embed) {
            fprintf(stderr, "%s: is %s really an image file?\n", __func__, image_path.c_str());
            return;
        }
    }
}

void md_free(struct moondream_context * ctx_md) {
    if (ctx_md->ctx_clip) {
        clip_free(ctx_md->ctx_clip);
        ctx_md->ctx_clip = NULL;
    }

    llama_free_model(ctx_md->model);
    llama_backend_free();
}

void moondream_infer(struct moondream_context *ctx_md, struct llama_context *ctx_llama, std::string& image_path, std::string& prompt, int32_t n_threads, std::string &result) {
    struct image_embed_result embed_result;

    load_image(ctx_md,image_path, prompt, n_threads, embed_result);
    if (!embed_result.embed) {
        return;
    }

    // process the prompt
    process_prompt(ctx_llama, embed_result.embed, ctx_md->gpt_params, embed_result.prompt, &result);

    llama_print_timings(ctx_llama);
    moondream_image_embed_free(embed_result.embed);
    md_free(ctx_md);
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            fprintf(stderr, "%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
    return true;
}

static const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (id == llama_token_eos(llama_get_model(ctx_llama))) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}


void process_prompt(struct llama_context * ctx_llama, struct moondream_image_embed * image_embed, gpt_params * params, const std::string & prompt, std::string *result) {
    int n_past = 0;

    const int max_tgt_len = params->n_predict < 0 ? 256 : params->n_predict;
    const bool add_bos = llama_should_add_bos_token(llama_get_model(ctx_llama));

    std::string system_prompt, user_prompt;
    size_t image_pos = prompt.find("<image>");
    if (image_pos != std::string::npos) {
        // new templating mode: Provide the full prompt including system message and use <image> as a placeholder for the image
        system_prompt = prompt.substr(0, image_pos);
        user_prompt = prompt.substr(image_pos + std::string("<image>").length());
        printf("system_prompt: %s\n", system_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llama, system_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llama, tmp[i]).c_str());
            }
        }
        printf("user_prompt: %s\n", user_prompt.c_str());
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llama, tmp[i]).c_str());
            }
        }
    } else {
        // llava-1.5 native mode
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\nUSER:";
        user_prompt = prompt + "\nASSISTANT:";
        if (params->verbose_prompt) {
            auto tmp = ::llama_tokenize(ctx_llama, user_prompt, true, true);
            for (int i = 0; i < (int) tmp.size(); i++) {
                printf("%6d -> '%s'\n", tmp[i], llama_token_to_piece(ctx_llama, tmp[i]).c_str());
            }
        }
    }

    eval_string(ctx_llama, system_prompt.c_str(), params->n_batch, &n_past, add_bos);
    moondream_eval_image_embed(ctx_llama, image_embed, params->n_batch, &n_past);
    eval_string(ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);

    // generate the response

    fprintf(stderr, "\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(params->sparams);
    // empty string
    if (result) {
        *result = "";
    }
    std::string buffer = "";
    for (int i = 0; i < max_tgt_len; i++) {
        const char * tmp = sample(ctx_sampling, ctx_llama, &n_past);
        buffer += tmp;
        if (strcmp(tmp, "</s>") == 0) break;
        if (strstr(buffer.c_str(), "<END>")) {
            // print the pre-string
            for (size_t j = 0; j < buffer.length() - 5; j++) {
                std::cout << buffer[j];
            }
            if (result) {
                result->append(buffer, buffer.length() - 5);
            }
            break;
        }
        if (!endsWith(buffer, "<") && !endsWith(buffer, "<END")) {
            if (result) {
                result->append(buffer);
            }
            printf("%s", buffer.c_str());
            fflush(stdout);
            buffer = "";
        }
    }

    llama_sampling_free(ctx_sampling);
    printf("\n");
}
