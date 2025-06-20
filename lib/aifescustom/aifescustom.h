#ifndef AIFESCUSTOM_H
#define AIFESCUSTOM_H

#include <stdbool.h>
#include "aifes.h"
#include "aiconfiguration.h"

bool aialgo_calc_loss_acc_model_f32(aiconfiguration_t *ctx, aimodel_t *model, float *loss_result,
                                    float *accuracy_result);

aiopti_t *build_model(aiconfiguration_t *ctx, aimodel_t *model);


void prune(aimodel_t *model, float percentage);

void custom_ailayer_dense_forward(ailayer_t *self);

void save_model(const aimodel_t *model, FILE *f);

void load_model(const aimodel_t *model, FILE *f);

void run_training_loop(aiconfiguration_t *ctx, aimodel_t *model, aiopti_t *optimizer,
                       FILE *x_train, FILE *y_train, FILE *x_test, FILE *y_test);

void run_evaluation(aiconfiguration_t *ctx, aimodel_t *model, FILE *x_test, FILE *y_test);

#endif //AIFESCUSTOM_H
