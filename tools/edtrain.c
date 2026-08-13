#include "ed_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg(int argc,char **argv,const char *key){int i;for(i=1;i+1<argc;++i)if(strcmp(argv[i],key)==0)return argv[i+1];return NULL;}
static int has_flag(int argc,char **argv,const char *key){int i;for(i=1;i<argc;++i)if(strcmp(argv[i],key)==0)return 1;return 0;}
static const char *sample_name(uint32_t mode){
    switch(mode){case ED_SAMPLE_FULL:return "full";case ED_SAMPLE_ZOOM:return "zoom";case ED_SAMPLE_MIXED:return "mixed";case ED_SAMPLE_MOSAIC:return "mosaic";case ED_SAMPLE_TILE:return "tile";case ED_SAMPLE_ADAPT:return "adapt";default:return "auto";}
}
static ed_storage storage_from_name(const char *name){
    if(!name||strcmp(name,"fp16")==0)return ED_STORAGE_FP16;
    if(strcmp(name,"fp32")==0)return ED_STORAGE_FP32;
    if(strcmp(name,"int8")==0)return ED_STORAGE_INT8;
    if(strcmp(name,"int4")==0)return ED_STORAGE_INT4;
    return (ed_storage)0;
}
static const char *storage_name(ed_storage storage){
    switch(storage){case ED_STORAGE_FP32:return "fp32";case ED_STORAGE_FP16:return "fp16";case ED_STORAGE_INT8:return "int8";case ED_STORAGE_INT4:return "int4";default:return "invalid";}
}
static const char *adapter_name(ed_head_adapter adapter){return adapter==ED_HEAD_ADAPTER_CROSS_LEVEL?"cross-level":"independent";}
static const char *init_name(ed_class_init_mode mode){
    switch(mode){case ED_INIT_BEST_SINGLE:return "single";case ED_INIT_EQUAL_BLEND:return "equal";case ED_INIT_AP_WEIGHTED:return "ap-weighted";case ED_INIT_PROTOTYPE:return "prototype";case ED_INIT_PROTOTYPE_RESIDUAL:return "proto-residual";default:return "auto";}
}
static const char *solver_name(ed_proto_solver solver){
    switch(solver){case ED_PROTO_DIAG:return "diag";case ED_PROTO_LDA:return "lda";default:return "ridge";}
}
static void print_sha256(const char *path){
    uint8_t *data=NULL,digest[32];size_t n=0,i;if(ed_read_entire_file(path,&data,&n)!=ED_OK)return;
    ed_sha256(data,n,digest);free(data);printf("output_sha256=");for(i=0;i<32;++i)printf("%02X",digest[i]);printf(" output_bytes=%llu peak_rss_bytes=%llu\n",(unsigned long long)n,(unsigned long long)ed_process_peak_rss());
}
static ed_model *clone_model(const ed_model *source){
    ed_model *copy;uint32_t i;
    if(!source)return NULL;copy=(ed_model*)calloc(1,sizeof(*copy));if(!copy)return NULL;
    copy->architecture=source->architecture;copy->precision=source->precision;copy->flags=source->flags;copy->class_count=source->class_count;copy->tensor_count=source->tensor_count;copy->metadata_bytes=source->metadata_bytes;memcpy(copy->source_sha256,source->source_sha256,sizeof(copy->source_sha256));
    copy->class_names=(char(*)[ED_CLASS_NAME_BYTES])malloc((size_t)copy->class_count*ED_CLASS_NAME_BYTES);
    copy->tensors=(ed_tensor*)calloc(copy->tensor_count,sizeof(*copy->tensors));copy->metadata=(char*)malloc((size_t)copy->metadata_bytes+1u);
    if(!copy->class_names||(!copy->tensors&&copy->tensor_count)||!copy->metadata){ed_model_free(copy);return NULL;}
    memcpy(copy->class_names,source->class_names,(size_t)copy->class_count*ED_CLASS_NAME_BYTES);
    if(copy->metadata_bytes)memcpy(copy->metadata,source->metadata,copy->metadata_bytes);copy->metadata[copy->metadata_bytes]=0;
    for(i=0;i<copy->tensor_count;++i){copy->tensors[i]=source->tensors[i];copy->tensors[i].data=malloc((size_t)source->tensors[i].data_bytes?source->tensors[i].data_bytes:1u);if(!copy->tensors[i].data){ed_model_free(copy);return NULL;}memcpy(copy->tensors[i].data,source->tensors[i].data,(size_t)source->tensors[i].data_bytes);}
    return copy;
}
int main(int argc,char **argv){
    const char *dataset_path=arg(argc,argv,"--dataset"),*selection_path=arg(argc,argv,"--selection-dataset"),*selection_score_text=arg(argc,argv,"--selection-score-threshold"),*weights_path=arg(argc,argv,"--weights"),*output_path=arg(argc,argv,"--output"),*storage_text=arg(argc,argv,"--output-precision"),*adapter_text=arg(argc,argv,"--head-adapter"),*cross_mix_text=arg(argc,argv,"--cross-level-mix"),*budget_text=arg(argc,argv,"--budget-ms"),*threads_text=arg(argc,argv,"--threads"),*seed_text=arg(argc,argv,"--seed"),*lr_text=arg(argc,argv,"--learning-rate"),*mosaic_text=arg(argc,argv,"--mosaic-size"),*accumulation_text=arg(argc,argv,"--accumulation"),*scope_text=arg(argc,argv,"--train-scope"),*cache_text=arg(argc,argv,"--feature-cache-samples"),*max_steps_text=arg(argc,argv,"--max-steps"),*restart_text=arg(argc,argv,"--restart-after-steps"),*sample_text=arg(argc,argv,"--sample-mode"),*ema_text=arg(argc,argv,"--ema-decay"),*wise_text=arg(argc,argv,"--wise-mix"),*dfl_text=arg(argc,argv,"--dfl-lr-scale"),*calibration_text=arg(argc,argv,"--calibration-samples"),*init_text=arg(argc,argv,"--init-mode"),*solver_text=arg(argc,argv,"--proto-solver"),*class_lr_text=arg(argc,argv,"--class-lr");
    ed_model *model=NULL,*source_guard=NULL,*guard_model=NULL;ed_dataset *dataset=NULL,*selection=NULL;ed_train_report report;ed_status status;uint64_t start=ed_monotonic_ms(),total_budget=50000u,used,reserve,calibration_ms=0,pipeline_elapsed=0;int calibrated=0,novel_classes=0,zero_update=has_flag(argc,argv,"--zero-update"),output_saved=0;
    float selection_score=selection_score_text?strtof(selection_score_text,NULL):0.01f;
    ed_class_init_mode init_mode=ED_INIT_EQUAL_BLEND;ed_proto_solver proto_solver=ED_PROTO_RIDGE;
    float class_lr_scale[ED_MAX_CLASSES],class_wise_scale[ED_MAX_CLASSES];ed_class_calibration_report cr;
    ed_train_config config={50000,1,1,ED_TRAIN_OUTPUTS,0,4,0.001f,0.0001f,0.9f,0.00004f,NULL,0,0,ED_SAMPLE_AUTO,0.0f,0.0f,1.0f,0,ED_STORAGE_FP16,NULL,NULL,0,0,ED_HEAD_ADAPTER_INDEPENDENT,0.0f,0,0,0,0,0};
    memset(&cr,0,sizeof(cr));memset(&report,0,sizeof(report));
    if(!dataset_path||!output_path||selection_score<0.0f||selection_score>1.0f){fprintf(stderr,"usage: edtrain --dataset D.edb [--selection-dataset V.edb --selection-score-threshold 0.01] [--weights models/picodet_s_coco80_fp16.edm] --output O.edm [--output-precision fp32|fp16|int8|int4] [--head-adapter independent|cross-level] [--cross-level-mix F] [--roi-head] [--picofeat-adapter] [--quality-adapter] [--nesterov] [--aligned-loss] [--budget-ms 50000] [--threads N] [--seed N] [--learning-rate LR] [--mosaic-size 0|1|4] [--accumulation N] [--train-scope outputs|classification|quality] [--feature-cache-samples N] [--max-steps N] [--restart-after-steps N] [--sample-mode auto|full|zoom|mixed|mosaic|tile|adapt] [--calibration-samples N] [--init-mode auto|single|equal|ap-weighted|prototype|proto-residual] [--proto-solver ridge|diag|lda] [--class-lr off|auto] [--class-loss-norm] [--hard-neg] [--zero-update] [--no-class-calibration] [--dfl-lr-scale F] [--ema-decay F] [--wise-mix F]\n");return 2;}
    if(!weights_path)weights_path="models/picodet_s_coco80_fp16.edm";
    if(storage_text){config.checkpoint_storage=storage_from_name(storage_text);if(!config.checkpoint_storage){fprintf(stderr,"invalid --output-precision (expected fp32, fp16, int8, or int4)\n");return 2;}}
    if(adapter_text){if(strcmp(adapter_text,"cross-level")==0){config.head_adapter=ED_HEAD_ADAPTER_CROSS_LEVEL;config.cross_level_mix=1.0f;}else if(strcmp(adapter_text,"independent")!=0){fprintf(stderr,"invalid --head-adapter (expected independent or cross-level)\n");return 2;}}
    if(cross_mix_text){config.head_adapter=ED_HEAD_ADAPTER_CROSS_LEVEL;config.cross_level_mix=strtof(cross_mix_text,NULL);if(config.cross_level_mix<=0.0f||config.cross_level_mix>1.0f){fprintf(stderr,"invalid --cross-level-mix (expected 0 < F <= 1)\n");return 2;}}
    if(init_text){
        if(strcmp(init_text,"auto")==0)init_mode=ED_INIT_AUTO;
        else if(strcmp(init_text,"single")==0)init_mode=ED_INIT_BEST_SINGLE;
        else if(strcmp(init_text,"equal")==0)init_mode=ED_INIT_EQUAL_BLEND;
        else if(strcmp(init_text,"ap-weighted")==0)init_mode=ED_INIT_AP_WEIGHTED;
        else if(strcmp(init_text,"prototype")==0)init_mode=ED_INIT_PROTOTYPE;
        else if(strcmp(init_text,"proto-residual")==0)init_mode=ED_INIT_PROTOTYPE_RESIDUAL;
        else{fprintf(stderr,"invalid --init-mode\n");return 2;}
    }
    if(solver_text){
        if(strcmp(solver_text,"ridge")==0)proto_solver=ED_PROTO_RIDGE;
        else if(strcmp(solver_text,"diag")==0)proto_solver=ED_PROTO_DIAG;
        else if(strcmp(solver_text,"lda")==0)proto_solver=ED_PROTO_LDA;
        else{fprintf(stderr,"invalid --proto-solver\n");return 2;}
    }
    if(has_flag(argc,argv,"--class-loss-norm"))config.class_loss_norm=1u;
    if(has_flag(argc,argv,"--hard-neg"))config.hard_neg_boost=1u;
    if(has_flag(argc,argv,"--nesterov"))config.nesterov=1u;
    if(has_flag(argc,argv,"--roi-head"))config.roi_head=1u;
    if(has_flag(argc,argv,"--picofeat-adapter"))config.picofeat_adapter=1u;
    if(has_flag(argc,argv,"--quality-adapter"))config.quality_adapter=1u;
    if(has_flag(argc,argv,"--aligned-loss"))config.aligned_loss=1u;
    if(budget_text)total_budget=strtoull(budget_text,NULL,10);if(threads_text)config.threads=(uint32_t)strtoul(threads_text,NULL,10);
    if(seed_text)config.seed=strtoull(seed_text,NULL,10);
    if(lr_text){config.learning_rate=strtof(lr_text,NULL);config.final_learning_rate=config.learning_rate*0.1f;}
    if(mosaic_text)config.mosaic_size=(uint32_t)strtoul(mosaic_text,NULL,10);
    if(accumulation_text)config.accumulation=(uint32_t)strtoul(accumulation_text,NULL,10);
    if(cache_text)config.feature_cache_samples=(uint32_t)strtoul(cache_text,NULL,10);
    if(max_steps_text)config.max_optimizer_steps=(uint32_t)strtoul(max_steps_text,NULL,10);
    if(restart_text)config.restart_after_steps=(uint32_t)strtoul(restart_text,NULL,10);
    if(ema_text)config.ema_decay=strtof(ema_text,NULL);if(wise_text)config.wise_mix=strtof(wise_text,NULL);
    if(scope_text){if(strcmp(scope_text,"classification")==0)config.scope=ED_TRAIN_CLASSIFICATION;else if(strcmp(scope_text,"quality")==0)config.scope=ED_TRAIN_QUALITY;else if(strcmp(scope_text,"outputs")!=0)return 2;}
    if(sample_text){
        if(strcmp(sample_text,"full")==0)config.sample_mode=ED_SAMPLE_FULL;
        else if(strcmp(sample_text,"zoom")==0)config.sample_mode=ED_SAMPLE_ZOOM;
        else if(strcmp(sample_text,"mixed")==0)config.sample_mode=ED_SAMPLE_MIXED;
        else if(strcmp(sample_text,"mosaic")==0)config.sample_mode=ED_SAMPLE_MOSAIC;
        else if(strcmp(sample_text,"tile")==0)config.sample_mode=ED_SAMPLE_TILE;
        else if(strcmp(sample_text,"adapt")==0)config.sample_mode=ED_SAMPLE_ADAPT;
        else if(strcmp(sample_text,"auto")!=0)return 2;
    }
    if(dfl_text)config.dfl_lr_scale=strtof(dfl_text,NULL);
    if(config.quality_adapter&&!config.aligned_loss){fprintf(stderr,"--quality-adapter requires --aligned-loss\n");return 2;}
    if(config.scope==ED_TRAIN_QUALITY&&!config.quality_adapter){fprintf(stderr,"--train-scope quality requires --quality-adapter\n");return 2;}
    if(!cache_text&&!has_flag(argc,argv,"--no-feature-cache")&&total_budget<=15000u){
        config.feature_cache_samples=12u;
        if(!max_steps_text)config.max_optimizer_steps=120u;
        if(!lr_text){config.learning_rate=0.000125f;config.final_learning_rate=0.0000125f;}
    }
    if((status=ed_dataset_load(dataset_path,&dataset))!=ED_OK||(status=ed_model_load(weights_path,&model))!=ED_OK){fprintf(stderr,"load: %s\n",ed_status_string(status));goto fail;}
    if(selection_path&&!zero_update){const char *names[ED_MAX_CLASSES];uint32_t i;
        source_guard=clone_model(model);if(!source_guard){status=ED_ERR_MEMORY;fprintf(stderr,"selection source guard: %s\n",ed_status_string(status));goto fail;}
        for(i=0;i<dataset->header.class_count;++i)names[i]=dataset->class_names[i];
        status=ed_model_remap_classes(source_guard,names,dataset->header.class_count);if(status!=ED_OK){fprintf(stderr,"selection source map: %s\n",ed_status_string(status));goto fail;}
    }
    if(model->class_count>dataset->header.class_count&&!has_flag(argc,argv,"--no-class-calibration")){
        char source_names[ED_MAX_CLASSES][ED_CLASS_NAME_BYTES];const char *names[ED_MAX_CLASSES];uint32_t i,samples,named=0;
        for(i=0;i<dataset->header.class_count;++i){names[i]=dataset->class_names[i];if(ed_find_source_class(model,dataset->class_names[i])>=0)++named;}
        if(named==dataset->header.class_count){
            printf("class_map=name-or-alias matched=%u\n",named);
            for(i=0;i<dataset->header.class_count;++i){int src=ed_find_source_class(model,dataset->class_names[i]);printf("class_map target=%s source=%s[%d]\n",dataset->class_names[i],src>=0?model->class_names[src]:"?",src);}
            status=ed_model_remap_classes(model,names,dataset->header.class_count);
            if(status!=ED_OK){fprintf(stderr,"class map: %s\n",ed_status_string(status));goto fail;}
            if(!cache_text&&!has_flag(argc,argv,"--no-feature-cache"))config.feature_cache_samples=16u;
            if(!max_steps_text)config.max_optimizer_steps=80u;
            if(!lr_text){config.learning_rate=0.00006f;config.final_learning_rate=0.000006f;}
            if(!wise_text)config.wise_mix=0.70f;
        }else{
            uint64_t used_ms=ed_monotonic_ms()-start,remain;uint32_t desired_samples=192u/dataset->header.class_count;
            if(desired_samples>64u)desired_samples=64u;if(desired_samples<4u)desired_samples=4u;
            samples=calibration_text?(uint32_t)strtoul(calibration_text,NULL,10):desired_samples;
            remain=total_budget>used_ms+8000u?total_budget-used_ms-8000u:0u;
            if(samples==0u){status=ED_ERR_ARGUMENT;fprintf(stderr,"class calibration: %s\n",ed_status_string(status));goto fail;}
            if(remain<1500u){
                printf("class_map=deadline-remap unmatched=%u\n",dataset->header.class_count-named);
                status=ed_model_remap_classes(model,names,dataset->header.class_count);
                if(status!=ED_OK){fprintf(stderr,"class map: %s\n",ed_status_string(status));goto fail;}
                if(!cache_text&&!has_flag(argc,argv,"--no-feature-cache"))config.feature_cache_samples=12u;
                if(!max_steps_text)config.max_optimizer_steps=120u;
                if(!lr_text){config.learning_rate=0.000125f;config.final_learning_rate=0.0000125f;}
            }else{
                uint64_t calibration_start=ed_monotonic_ms();int calibration_ready=0;
                for(i=0;i<model->class_count;++i)strcpy(source_names[i],model->class_names[i]);
                if(!calibration_text){
                    ed_class_calibration_report probe;uint64_t probe_start=ed_monotonic_ms(),probe_ms,calibration_budget=total_budget/4u,units=1u;
                    memset(&probe,0,sizeof(probe));status=ed_model_calibrate_classes_ex(model,dataset,1u,config.seed,init_mode,proto_solver,&probe);probe_ms=ed_monotonic_ms()-probe_start;
                    if(status!=ED_OK){fprintf(stderr,"calibration probe: %s\n",ed_status_string(status));goto fail;}
                    if(probe_ms==0u)probe_ms=1u;if(calibration_budget>remain)calibration_budget=remain;
                    if(calibration_budget>probe_ms)units=(calibration_budget-probe_ms)/probe_ms;
                    if(units>desired_samples)units=desired_samples;
                    samples=units>=2u?(uint32_t)units:1u;
                    printf("calibration_probe_ms=%llu calibration_probe_records=%u calibration_budget_ms=%llu samples_per_class=%u\n",(unsigned long long)probe_ms,probe.calibration_records,(unsigned long long)calibration_budget,samples);
                    if(samples==1u){cr=probe;calibration_ready=1;}
                    else{ed_model_free(model);model=NULL;status=ed_model_load(weights_path,&model);if(status!=ED_OK){fprintf(stderr,"calibration reload: %s\n",ed_status_string(status));goto fail;}}
                }
                if(!calibration_ready)status=ed_model_calibrate_classes_ex(model,dataset,samples,config.seed,init_mode,proto_solver,&cr);
                if(status!=ED_OK){fprintf(stderr,"class calibration: %s\n",ed_status_string(status));goto fail;}
                calibration_ms=ed_monotonic_ms()-calibration_start;calibrated=1;printf("init_mode=%s proto_solver=%s fold_map50=%.6f samples=%u calibration_records=%u calibration_ms=%llu\n",init_name((ed_class_init_mode)cr.init_mode_used),solver_name((ed_proto_solver)cr.proto_solver_used),cr.fold_map50,samples,cr.calibration_records,(unsigned long long)calibration_ms);
                for(i=0;i<cr.class_count;++i){uint32_t k;if(cr.source_count[i]>1u)novel_classes=1;printf("class_calibration target=%s sources=",dataset->class_names[i]);for(k=0;k<cr.source_count[i];++k)printf("%s%s[%u:%.6f]",k?",":"",source_names[cr.source_class[i][k]],cr.source_class[i][k],cr.source_affinity[i][k]);printf(" blend_affinity=%.6f objects=%u proto_pos=%u proto_neg=%u fold_ap=%.6f norm=%.4f\n",cr.blended_affinity[i],cr.calibrated_objects[i],cr.proto_positives[i],cr.proto_negatives[i],cr.fold_ap[i],cr.classifier_norm[i]);}
            }
        }
    }else{const char *names[ED_MAX_CLASSES];uint32_t i;for(i=0;i<dataset->header.class_count;++i)names[i]=dataset->class_names[i];status=ed_model_remap_classes(model,names,dataset->header.class_count);if(status!=ED_OK){fprintf(stderr,"class map: %s\n",ed_status_string(status));goto fail;}
        if(!cache_text&&!has_flag(argc,argv,"--no-feature-cache")&&!config.feature_cache_samples){config.feature_cache_samples=12u;if(!max_steps_text)config.max_optimizer_steps=120u;if(!lr_text){config.learning_rate=0.000125f;config.final_learning_rate=0.0000125f;}}}
    if(calibrated){
        if(novel_classes){if(!scope_text)config.scope=ED_TRAIN_CLASSIFICATION;if(!cache_text&&!has_flag(argc,argv,"--no-feature-cache"))config.feature_cache_samples=64u;if(!max_steps_text)config.max_optimizer_steps=360u;if(!lr_text){config.learning_rate=0.00025f;config.final_learning_rate=0.000025f;}if(!wise_text)config.wise_mix=0.50f;if(!restart_text&&!max_steps_text)config.restart_after_steps=240u;if(!has_flag(argc,argv,"--no-class-loss-norm"))config.class_loss_norm=1u;if(!has_flag(argc,argv,"--no-hard-neg"))config.hard_neg_boost=1u;}
        else{if(!cache_text&&!has_flag(argc,argv,"--no-feature-cache"))config.feature_cache_samples=12u;if(!max_steps_text)config.max_optimizer_steps=120u;if(!lr_text){config.learning_rate=0.000125f;config.final_learning_rate=0.0000125f;}}
        if((!class_lr_text&&novel_classes)||(class_lr_text&&strcmp(class_lr_text,"auto")==0)){
            uint32_t i;for(i=0;i<cr.class_count;++i)class_lr_scale[i]=cr.source_affinity[i][0]>=0.60f?0.35f:1.75f;
            config.class_lr_scale=class_lr_scale;
        }
        if(has_flag(argc,argv,"--wise-class")){
            uint32_t i;for(i=0;i<cr.class_count;++i)class_wise_scale[i]=cr.source_affinity[i][0]>=0.60f?0.35f:0.70f;
            config.class_wise_mix=class_wise_scale;
        }
    }
    used=ed_monotonic_ms()-start;reserve=total_budget/50u;if(reserve<1000u)reserve=1000u;if(used+reserve>=total_budget){status=ED_ERR_DEADLINE;fprintf(stderr,"train: %s\n",ed_status_string(status));goto fail;}
    config.budget_ms=total_budget-used-reserve;config.checkpoint_prefix=output_path;
    if(selection_path&&!zero_update){guard_model=clone_model(model);if(!guard_model){status=ED_ERR_MEMORY;fprintf(stderr,"selection guard: %s\n",ed_status_string(status));goto fail;}}
    if(!zero_update){status=ed_train(model,dataset,&config,&report);if(status!=ED_OK){fprintf(stderr,"train: %s\n",ed_status_string(status));goto fail;}}
    pipeline_elapsed=ed_monotonic_ms()-start;report.elapsed_ms=pipeline_elapsed;report.deadline_respected=report.elapsed_ms<=total_budget;
    if(guard_model){ed_map_report source_map,baseline_map,adapted_map;ed_model *serialized=NULL;uint32_t ci,best=0;float best_map;
        status=ed_dataset_load(selection_path,&selection);if(status!=ED_OK){fprintf(stderr,"selection load: %s\n",ed_status_string(status));goto fail;}
        {uint64_t save_start=ed_monotonic_ms();status=ed_model_save_storage(model,output_path,config.checkpoint_storage);pipeline_elapsed+=ed_monotonic_ms()-save_start;}
        if(status==ED_OK)status=ed_model_load(output_path,&serialized);
        if(status==ED_OK)status=ed_evaluate_map50(source_guard,selection,selection_score,0.42f,&source_map);
        if(status==ED_OK)status=ed_evaluate_map50(guard_model,selection,selection_score,0.42f,&baseline_map);
        if(status==ED_OK)status=ed_evaluate_map50(serialized,selection,selection_score,0.42f,&adapted_map);
        if(status!=ED_OK){fprintf(stderr,"selection eval: %s\n",ed_status_string(status));goto fail;}
        best_map=source_map.map50;
        {int eligible=1;for(ci=0;ci<source_map.class_count;++ci)if(baseline_map.per_class_ap[ci]+1e-7f<source_map.per_class_ap[ci])eligible=0;if(eligible&&baseline_map.map50>best_map){best=1;best_map=baseline_map.map50;}}
        {int eligible=1;for(ci=0;ci<source_map.class_count;++ci)if(adapted_map.per_class_ap[ci]+1e-7f<source_map.per_class_ap[ci])eligible=0;if(eligible&&adapted_map.map50>best_map){best=2;best_map=adapted_map.map50;}}
        printf("selection_score_threshold=%.6f selection_source_map50=%.6f selection_calibrated_map50=%.6f selection_adapted_map50=%.6f selection_decision=%s\n",selection_score,source_map.map50,baseline_map.map50,adapted_map.map50,best==2?"adapted":(best==1?"calibrated":"source"));
        if(best==0){ed_model_free(serialized);ed_model_free(guard_model);guard_model=NULL;ed_model_free(model);model=source_guard;source_guard=NULL;}
        else if(best==1){ed_model_free(serialized);ed_model_free(source_guard);source_guard=NULL;ed_model_free(model);model=guard_model;guard_model=NULL;}
        else{ed_model_free(source_guard);source_guard=NULL;ed_model_free(guard_model);guard_model=NULL;ed_model_free(model);model=serialized;serialized=NULL;output_saved=1;}
    }
    if(!output_saved){uint64_t save_start=ed_monotonic_ms();status=ed_model_save_storage(model,output_path,config.checkpoint_storage);pipeline_elapsed+=ed_monotonic_ms()-save_start;}
    if(status!=ED_OK){fprintf(stderr,"save: %s\n",ed_status_string(status));goto fail;}report.elapsed_ms=pipeline_elapsed;report.deadline_respected=report.elapsed_ms<=total_budget;
    printf("elapsed_ms=%llu images=%llu mosaics=%llu steps=%llu trainable_parameters=%llu final_loss=%.6f deadline_respected=%d sample_mode=%s median_box_side=%.2f cache_samples=%u cache_ms=%llu calibration_ms=%llu output_precision=%s head_adapter=%s cross_level_mix=%.3f roi_head=%u picofeat_adapter=%u quality_adapter=%u aligned_loss=%u nesterov=%u assign_recall=%.4f assign_mean_iou=%.4f\n",(unsigned long long)report.elapsed_ms,(unsigned long long)report.images_seen,(unsigned long long)report.mosaics_seen,(unsigned long long)report.optimizer_steps,(unsigned long long)report.trainable_parameters,report.final_loss,report.deadline_respected,sample_name(report.sample_mode_used),report.median_box_side,report.feature_cache_samples,(unsigned long long)report.feature_cache_ms,(unsigned long long)calibration_ms,storage_name(config.checkpoint_storage),adapter_name(config.head_adapter),config.cross_level_mix,config.roi_head,config.picofeat_adapter,config.quality_adapter,config.aligned_loss,config.nesterov,report.assign_recall,report.assign_mean_iou);
    print_sha256(output_path);
    ed_dataset_free(selection);ed_dataset_free(dataset);ed_model_free(source_guard);ed_model_free(guard_model);ed_model_free(model);return report.deadline_respected?0:1;
fail:ed_dataset_free(selection);ed_dataset_free(dataset);ed_model_free(source_guard);ed_model_free(guard_model);ed_model_free(model);return 1;
}
