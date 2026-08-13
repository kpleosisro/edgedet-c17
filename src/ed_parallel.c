#include "ed_internal.h"
#include <stdlib.h>
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HANDLE native_thread;typedef CRITICAL_SECTION native_mutex;typedef CONDITION_VARIABLE native_cond;
#else
#include <pthread.h>
typedef pthread_t native_thread;typedef pthread_mutex_t native_mutex;typedef pthread_cond_t native_cond;
#endif
#define ED_MAX_WORKERS 63u
typedef struct worker_arg{uint32_t index;}worker_arg;
typedef struct{native_thread thread[ED_MAX_WORKERS];worker_arg arg[ED_MAX_WORKERS];native_mutex mutex;native_cond ready,done;ed_parallel_fn function;void*context;size_t count;uint64_t generation;uint32_t created,active,completed;int initialized;}parallel_pool;
static parallel_pool pool;static uint32_t requested_threads=1u;
#if defined(_WIN32)
static void lock(native_mutex*m){EnterCriticalSection(m);}static void unlock(native_mutex*m){LeaveCriticalSection(m);}static void waitc(native_cond*c,native_mutex*m){SleepConditionVariableCS(c,m,INFINITE);}static void wakeall(native_cond*c){WakeAllConditionVariable(c);}static void wakeone(native_cond*c){WakeConditionVariable(c);}static DWORD WINAPI worker_main(void*p)
#else
static void lock(native_mutex*m){pthread_mutex_lock(m);}static void unlock(native_mutex*m){pthread_mutex_unlock(m);}static void waitc(native_cond*c,native_mutex*m){pthread_cond_wait(c,m);}static void wakeall(native_cond*c){pthread_cond_broadcast(c);}static void wakeone(native_cond*c){pthread_cond_signal(c);}static void*worker_main(void*p)
#endif
{worker_arg*a=(worker_arg*)p;uint64_t seen;lock(&pool.mutex);seen=pool.generation;for(;;){while(seen==pool.generation)waitc(&pool.ready,&pool.mutex);seen=pool.generation;{ed_parallel_fn fn=pool.function;void*ctx=pool.context;size_t n=pool.count;uint32_t active=pool.active,index=a->index;unlock(&pool.mutex);if(index<active){size_t q=n/active,r=n%active,begin=q*index+(index<r?index:r),end=begin+q+(index<r?1u:0u);fn(ctx,begin,end);}lock(&pool.mutex);}++pool.completed;if(pool.completed==pool.created)wakeone(&pool.done);}
#if defined(_WIN32)
return 0;
#else
return NULL;
#endif
}
static int initialize(uint32_t workers){uint32_t i;if(!pool.initialized){memset(&pool,0,sizeof(pool));
#if defined(_WIN32)
InitializeCriticalSection(&pool.mutex);InitializeConditionVariable(&pool.ready);InitializeConditionVariable(&pool.done);
#else
if(pthread_mutex_init(&pool.mutex,NULL)||pthread_cond_init(&pool.ready,NULL)||pthread_cond_init(&pool.done,NULL))return 0;
#endif
pool.initialized=1;}for(i=pool.created;i<workers;++i){pool.arg[i].index=i+1u;
#if defined(_WIN32)
pool.thread[i]=CreateThread(NULL,0,worker_main,&pool.arg[i],0,NULL);if(!pool.thread[i])return 0;
#else
if(pthread_create(&pool.thread[i],NULL,worker_main,&pool.arg[i]))return 0;
#endif
++pool.created;}return 1;}
ed_status ed_runtime_set_threads(uint32_t n){if(n<1u||n>64u)return ED_ERR_ARGUMENT;if(!initialize(n-1u))return ED_ERR_MEMORY;requested_threads=n;return ED_OK;}
uint32_t ed_runtime_threads(void){return requested_threads;}
ed_status ed_parallel_for(size_t count,ed_parallel_fn fn,void*ctx){uint32_t active=requested_threads;if(!fn)return ED_ERR_ARGUMENT;if(count==0)return ED_OK;if(active<=1u||count<2u){fn(ctx,0,count);return ED_OK;}if(active>count)active=(uint32_t)count;if(!initialize(active-1u))return ED_ERR_MEMORY;lock(&pool.mutex);pool.function=fn;pool.context=ctx;pool.count=count;pool.active=active;pool.completed=0;++pool.generation;wakeall(&pool.ready);unlock(&pool.mutex);{size_t q=count/active,r=count%active,end=q+(r?1u:0u);fn(ctx,0,end);}lock(&pool.mutex);while(pool.completed!=pool.created)waitc(&pool.done,&pool.mutex);unlock(&pool.mutex);return ED_OK;}
