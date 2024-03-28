#include "ggml.h"
#include "common.h"
#include "vision_encoder.h"
#include "moondream.h"


static void show_additional_info(int /*argc*/, char ** argv) {
    fprintf(stderr, "\n example usage: %s -m <moondream2/ggml-model-q5_k.gguf> --mmproj <moondream2/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    fprintf(stderr, "  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

int main(int argc, char ** argv) {
    ggml_time_init();

    if (argc > 4 && strcmp(argv[1], "quantize") == 0) {
        char *fname_input = argv[2];
        char *fname_ouput = argv[3];
        ggml_type type = GGML_TYPE_Q4_1;
        clip_model_quantize(fname_input, fname_ouput, type);
        return 1;
    }

    struct gpt_params gpt_params;

    if (!gpt_params_parse(argc, argv, gpt_params)) {
        show_additional_info(argc, argv);
        return 1;
    }

    if (gpt_params.mmproj.empty() || (gpt_params.image.empty() && !prompt_contains_image(gpt_params.prompt))) {
        gpt_print_usage(argc, argv, gpt_params);
        show_additional_info(argc, argv);
        return 1;
    }

    auto ctx_md = moondream_init(gpt_params);
    if (ctx_md == NULL) {
        fprintf(stderr, "%s: error: failed to init moondream\n", __func__);
        return 1;
    }

    struct image_embed_result image_embed_result;
    load_image(ctx_md,gpt_params.image, gpt_params.prompt, gpt_params.n_threads, image_embed_result);
    if (!image_embed_result.embed) {
        return 1;
    }

    struct llama_context *ctx_llama = new_llama_context(ctx_md);

    // process the prompt
    process_prompt(ctx_llama, image_embed_result.embed, &gpt_params, image_embed_result.prompt, nullptr);

    llama_print_timings(ctx_llama);

    moondream_image_embed_free(image_embed_result.embed);
    llama_free(ctx_llama);
    md_free(ctx_md);
    return 0;
}
