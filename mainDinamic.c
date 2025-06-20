#include <string.h>
#include <time.h>

#include "log.h"
#include "csv.h"
#include "aifes.h"
#include "memmanager.h"
#include "aifescustom.h"
#include "aiconfiguration.h"


int main(int argc, char *argv[]) {
    unsigned int seed = (unsigned int) time(NULL) ^ (uintptr_t) &seed;
    srand(seed);

    FILE *x_train = NULL, *y_train = NULL, *x_test = NULL, *y_test = NULL;
    FILE *save = NULL, *load = NULL;

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

    if (!open_csv(&x_train, ctx.basedir, "x_train.csv", "r")
        || !open_csv(&y_train, ctx.basedir, "y_train.csv", "r")
        || !open_csv(&x_test, ctx.basedir, "x_test.csv", "r")
        || !open_csv(&y_test, ctx.basedir, "y_test.csv", "r")) {
        SAFE_EXIT_FAILURE("Errore apertura file CSV");
    }

    if (ctx.load) {
        open_csv(&load, ctx.basedir, ctx.load, "r");
        load_model(&model, load);
    }

    if (ctx.training) {
        run_training_loop(&ctx, &model, optimizer, x_train, y_train, x_test, y_test);
    } else {
        run_evaluation(&ctx, &model, x_test, y_test);
    }

    if (ctx.save) {
        open_csv(&save, ctx.basedir, ctx.save, "w");
        save_model(&model, save);
    }

    CLOSE_ALL_FILES(x_train, y_train, x_test, y_test, save, load);
    LOG_INFO("Freed %llu byte", mem_total());
    SAFE_EXIT_SUCCESS("");
}
