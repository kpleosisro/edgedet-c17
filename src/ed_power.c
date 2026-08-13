#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "ed_internal.h"
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#endif

static float power_limit_w=0.0f;
static float busy_core_w=8.0f;
static float last_avg_w=0.0f;
static float last_sleep_ms=0.0f;
static uint64_t mark_cpu_ns=0,mark_wall_ns=0;

static uint64_t process_cpu_ns(void){
#if defined(_WIN32)
    FILETIME create,exit_t,kernel,user;ULARGE_INTEGER k,u;
    if(!GetProcessTimes(GetCurrentProcess(),&create,&exit_t,&kernel,&user))return 0;
    k.LowPart=kernel.dwLowDateTime;k.HighPart=kernel.dwHighDateTime;
    u.LowPart=user.dwLowDateTime;u.HighPart=user.dwHighDateTime;
    return (k.QuadPart+u.QuadPart)*100ull;
#else
    struct timespec ts;
    if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&ts)!=0)return 0;
    return (uint64_t)ts.tv_sec*UINT64_C(1000000000)+(uint64_t)ts.tv_nsec;
#endif
}
static void sleep_ns(uint64_t ns){
#if defined(_WIN32)
    DWORD ms=(DWORD)((ns+999999ull)/1000000ull);if(ms)Sleep(ms);
#else
    struct timespec ts;ts.tv_sec=(time_t)(ns/UINT64_C(1000000000));ts.tv_nsec=(long)(ns%UINT64_C(1000000000));
    while(nanosleep(&ts,&ts)==-1){}
#endif
}
static void pin_one_cpu(void){
#if defined(_WIN32)
    SetProcessAffinityMask(GetCurrentProcess(),(DWORD_PTR)1);
#else
    cpu_set_t set;CPU_ZERO(&set);CPU_SET(0,&set);(void)sched_setaffinity(0,sizeof(set),&set);
#endif
}

ed_status ed_runtime_set_power_limit_w(float watts){
    if(watts<0.0f)return ED_ERR_ARGUMENT;
    power_limit_w=watts;last_avg_w=0.0f;last_sleep_ms=0.0f;
    if(watts>0.0f){pin_one_cpu();if(ed_runtime_threads()>1u)(void)ed_runtime_set_threads(1u);}
    return ED_OK;
}
float ed_runtime_power_limit_w(void){return power_limit_w;}
ed_status ed_runtime_set_busy_core_watts(float watts){
    if(watts<=0.0f||watts>200.0f)return ED_ERR_ARGUMENT;busy_core_w=watts;return ED_OK;
}
float ed_runtime_busy_core_watts(void){return busy_core_w;}
float ed_runtime_last_average_w(void){return last_avg_w;}
float ed_runtime_last_sleep_ms(void){return last_sleep_ms;}

void ed_runtime_power_mark_begin(void){
    mark_cpu_ns=process_cpu_ns();mark_wall_ns=ed_monotonic_ns();
}
void ed_runtime_power_mark_end(void){
    uint64_t cpu,wall,need_ns;double energy_j,wall_s;
    if(power_limit_w<=0.0f||mark_wall_ns==0u)return;
    cpu=process_cpu_ns()-mark_cpu_ns;wall=ed_monotonic_ns()-mark_wall_ns;
    if(wall==0u)wall=1u;
    energy_j=((double)cpu/1.0e9)* (double)busy_core_w;
    wall_s=(double)wall/1.0e9;
    last_avg_w=(float)(energy_j/wall_s);
    need_ns=(uint64_t)((energy_j/(double)power_limit_w)*1.0e9+0.5);
    last_sleep_ms=0.0f;
    if(need_ns>wall){
        uint64_t extra=need_ns-wall;sleep_ns(extra);last_sleep_ms=(float)extra/1.0e6f;
        last_avg_w=(float)(energy_j/((double)need_ns/1.0e9));
    }
    mark_wall_ns=0;
}
void ed_runtime_power_apply(void){ed_runtime_power_mark_end();}
