#ifndef AIFESCUSTOM_H
#define AIFESCUSTOM_H

#include "aifes.h"
#include "aiconfiguration.h"

extern FILE *x_train, *y_train, *x_test, *y_test;
extern FILE *save, *load;

aiopti_t *build_model(aiconfiguration_t *ctx, aimodel_t *model);

void save_model(const aimodel_t *model, FILE *f);

void load_model(const aimodel_t *model, FILE *f);

void run_training(aiconfiguration_t *ctx, aimodel_t *model, aiopti_t *optimizer);

void run_evaluation(aiconfiguration_t *ctx, aimodel_t *model);

#endif //AIFESCUSTOM_H
