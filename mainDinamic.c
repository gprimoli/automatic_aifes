#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include "log.h"
#include "csv.h"
#include "main.h"
#include "memmanager.h"
#include "aifescustom.h"
#include "aiconfiguration.h"
#include "aifescustom_internal.h"

int main(int argc, char *argv[]) {
    unsigned int seed = get_seed();
    srand(seed);
    LOG_INFO("Seed: %u", seed);

    aiconfiguration_t *conf = mem_calloc(1, sizeof(aiconfiguration_t));
    aimodel_t *model = mem_calloc(1, sizeof(aimodel_t));

    if (argc != 2) {
        SAFE_EXIT_FAILURE("Inserire path configurazione");
    }

    if (load_config(conf, argv[1]) != 0) {
        SAFE_EXIT_FAILURE("Errore file configurazione");
    }

    aiopti_t *optimizer = build_model(conf, model);
    if (!optimizer) {
        SAFE_EXIT_FAILURE("Errore costruzione modello");
    }

    aiprint("\n-------------- Model structure ---------------\n");
    aialgo_print_model_structure(model);
    aiprint("----------------------------------------------\n\n");

    if (conf->load) {
        load_model(conf, model);
    }

    if (conf->training) {
        run_training(conf, model, optimizer);
    }

    if (conf->save) {
        save_model(conf, model);
    }

    if (conf->quantization != UNKNOWN_QUANTIZATION) {
        quantize(model, conf->quantization);
    }

    run_evaluation(conf, model);

    LOG_INFO("Batch size: %d", conf->batch_size);
    LOG_INFO("Pruning: %f%%", conf->pruning);
    LOG_INFO("Memoria allocata: %.4f KB", mem_total() / 1024.0);

    SAFE_EXIT_SUCCESS("Finish: ALL RIGHT");
}


unsigned int get_seed() {
    struct timeval tv;
    gettimeofday(&tv, NULL);

    pid_t pid = getpid();

    unsigned int entropy = (unsigned int) (tv.tv_sec ^ tv.tv_usec ^ pid ^ (uintptr_t) &tv);

    return entropy;
}
