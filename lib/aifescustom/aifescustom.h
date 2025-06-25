#ifndef AIFESCUSTOM_H
#define AIFESCUSTOM_H

#include "aifes.h"
#include "aiconfiguration.h"

aiopti_t *build_model(aiconfiguration_t *ctx, aimodel_t *model);

void save_model(const aimodel_t *model, FILE *f);

void load_model(const aimodel_t *model, FILE *f);

void run_training(aiconfiguration_t *ctx, aimodel_t *model, aiopti_t *optimizer,
                  FILE *x_train, FILE *y_train, FILE *x_test, FILE *y_test);

void run_evaluation(aiconfiguration_t *ctx, aimodel_t *model, FILE *x_test, FILE *y_test);

#endif //AIFESCUSTOM_H
