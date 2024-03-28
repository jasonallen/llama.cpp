#ifndef MOONDREAM_H
#define MOONDREAM_H

#include "ggml.h"
#include "common/common.h"
#include <vector>


#ifdef MOONDREAM_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef MOONDREAM_BUILD
#            define MOONDREAM_API __declspec(dllexport)
#        else
#            define MOONDREAM_API __declspec(dllimport)
#        endif
#    else
#        define MOONDREAM_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MOONDREAM_API
#endif // MOONDREAM_SHARED

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
#endif

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};


struct moondream_image_embed {
    float * embed;
    int n_image_pos;
};

struct  image_embed_result {
    struct moondream_image_embed * embed;
    std::string prompt;
};

struct moondream_context {
    struct clip_ctx * ctx_clip = NULL;
    struct gpt_params *gpt_params;
    struct llama_model * model = NULL;
};


/** sanity check for clip <-> moondream embed size match */
MOONDREAM_API bool moondream_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip);

MOONDREAM_API bool moondream_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out);

/** build an image embed from image file bytes */
MOONDREAM_API struct moondream_image_embed * moondream_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length);
/** build an image embed from a path to an image filename */
MOONDREAM_API struct moondream_image_embed * moondream_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path);

/** free an embedding made with moondream_image_embed_make_* */
MOONDREAM_API void moondream_image_embed_free(struct moondream_image_embed * embed);

/** write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past. on completion, n_past points to the next position in the context after the image embed. */
MOONDREAM_API bool moondream_eval_image_embed(struct llama_context * ctx_llama, const struct moondream_image_embed * embed, int n_batch, int * n_past);

/** creates a moondream context initialized with the vision and text models specified in params*/
MOONDREAM_API struct moondream_context * moondream_init(struct gpt_params &params);

/** returns a llama_context to prepare for inference */
MOONDREAM_API struct llama_context * new_llama_context(moondream_context* md_ctx);

/** generates an image embedding, may alter prompt  */
MOONDREAM_API void load_image(moondream_context * ctx_md, std::string& image_path, std::string& prompt, int32_t n_threads, struct image_embed_result &result);

/** prompts can contain images embedded in them */
MOONDREAM_API bool prompt_contains_image(std::string prompt);

/** infers a prompt */
MOONDREAM_API void process_prompt(struct llama_context * ctx_llama, struct moondream_image_embed * image_embed, gpt_params * params, const std::string & prompt, std::string *result);

/** frees the moondream_context */
MOONDREAM_API void md_free(struct moondream_context * ctx_md);

/** frees an image_embed_result */
MOONDREAM_API void free_image_embed_result(struct image_embed_result * image_embed_result);

/** runs inference on the image and prompt */
MOONDREAM_API void moondream_infer(struct moondream_context *ctx_md, struct llama_context *ctx_llama, std::string& image_path, std::string& prompt, int32_t n_threads, std::string &result);

#ifdef __cplusplus
}
#endif

#endif // MOONDREAM_H
