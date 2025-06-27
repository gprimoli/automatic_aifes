#include <string.h>
#include <time.h>

#include "log.h"
#include "csv.h"
#include "main.h"
#include "memmanager.h"
#include "aifescustom.h"
#include "aiconfiguration.h"

int main(int argc, char *argv[]) {
    unsigned int seed = get_seed();
    srand(seed);
    LOG_INFO("Seed: %u", seed);

    aiconfiguration_t ctx = {0};
    aimodel_t model = {0};

    if (argc != 2) {
        SAFE_EXIT_FAILURE("Inserire path configurazione");
    }

    if (load_config(argv[1], &ctx) != 0) {
        SAFE_EXIT_FAILURE("Errore file configurazione");
    }

    aiopti_t *optimizer = build_model(&ctx, &model);
    if (!optimizer) {
        SAFE_EXIT_FAILURE("Errore costruzione modello");
    }

    aiprint("\n-------------- Model structure ---------------\n");
    aialgo_print_model_structure(&model);
    aiprint("----------------------------------------------\n\n");

    if (ctx.load) {
        open_csv(&load, ctx.basedir, ctx.load, "r");
        load_model(&model, load);
    }

    if (ctx.training) {
        run_training(&ctx, &model, optimizer);
    } else {
        run_evaluation(&ctx, &model);
    }

    if (ctx.save) {
        open_csv(&save, ctx.basedir, ctx.save, "w");
        save_model(&model, save);
    }

    LOG_INFO("Batch size: %d", ctx.batch_size);
    LOG_INFO("Pruning: %f%%", ctx.pruning);
    LOG_INFO("Memoria allocata: %llu byte", mem_total());

    SAFE_EXIT_SUCCESS("Finish: ALL RIGHT");
}


unsigned int get_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    pid_t pid = getpid();

    unsigned int entropy = (unsigned int) (tv.tv_sec ^ tv.tv_usec ^ pid ^ (uintptr_t) &tv);

    return entropy;
}
