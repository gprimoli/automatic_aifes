#ifndef AIFESCUSTOM_H
#define AIFESCUSTOM_H

#include "aifes.h"
#include "csv_internal.h"
#include "aiconfiguration.h"

extern FILE *x_train, *y_train, *x_test, *y_test;

aiopti_t *build_model(aiconfiguration_t *conf, aimodel_t *model);

void save_model(aiconfiguration_t *conf, aimodel_t *model);

void load_model(aiconfiguration_t *conf, aimodel_t *model);

void run_training(aiconfiguration_t *conf, aimodel_t *model, aiopti_t *optimizer);

void run_evaluation(aiconfiguration_t *conf, aimodel_t *model);

#endif //AIFESCUSTOM_H
