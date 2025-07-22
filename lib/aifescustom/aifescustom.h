#ifndef AIFESCUSTOM_H
#define AIFESCUSTOM_H

#include "aifes.h"
#include "aiconfiguration.h"

// extern FILE *f_x_train_set,         *f_y_train_set;
// extern FILE *f_x_validation_set,    *f_y_validation_set;
// extern FILE *f_x_test_set,          *f_y_test_set;
// extern FILE *f_x_deployment_set,    *f_y_deployment_set;

aiopti_t *build_model(aiconfiguration_t *conf, aimodel_t *model);

void save_model(aiconfiguration_t *conf, aimodel_t *model);

void load_model(aiconfiguration_t *conf, aimodel_t *model);

void run_training(aiconfiguration_t *conf, aimodel_t *model, aiopti_t *optimizer);

void run_evaluation(aiconfiguration_t *conf, aimodel_t *model);

bool calc_scale_and_zero_point(Quantization qType, const aitensor_t *t, void *qParam);

#endif //AIFESCUSTOM_H
