#include "ed_internal.h"
#include <string.h>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

ed_status ed_mapped_file_open(const char *path,ed_mapped_file *file){
    if(!path||!file)return ED_ERR_ARGUMENT;memset(file,0,sizeof(*file));
#if defined(_WIN32)
    {
        HANDLE h,m;LARGE_INTEGER size;void *data;
        h=CreateFileA(path,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
        if(h==INVALID_HANDLE_VALUE)return ED_ERR_IO;
        if(!GetFileSizeEx(h,&size)||size.QuadPart<=0||(uint64_t)size.QuadPart>SIZE_MAX){CloseHandle(h);return ED_ERR_IO;}
        m=CreateFileMappingA(h,NULL,PAGE_READONLY,0,0,NULL);if(!m){CloseHandle(h);return ED_ERR_IO;}
        data=MapViewOfFile(m,FILE_MAP_READ,0,0,0);if(!data){CloseHandle(m);CloseHandle(h);return ED_ERR_IO;}
        file->data=(const uint8_t*)data;file->size=(size_t)size.QuadPart;file->file_handle=h;file->mapping_handle=m;return ED_OK;
    }
#else
    {
        int fd=open(path,O_RDONLY);struct stat st;void *data;
        if(fd<0)return ED_ERR_IO;if(fstat(fd,&st)!=0||st.st_size<=0||(uint64_t)st.st_size>SIZE_MAX){close(fd);return ED_ERR_IO;}
        data=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(data==MAP_FAILED){close(fd);return ED_ERR_IO;}
        file->data=(const uint8_t*)data;file->size=(size_t)st.st_size;file->file_handle=(void*)(intptr_t)(fd+1);return ED_OK;
    }
#endif
}
void ed_mapped_file_close(ed_mapped_file *file){
    if(!file)return;
#if defined(_WIN32)
    if(file->data)UnmapViewOfFile(file->data);if(file->mapping_handle)CloseHandle((HANDLE)file->mapping_handle);if(file->file_handle)CloseHandle((HANDLE)file->file_handle);
#else
    if(file->data)munmap((void*)file->data,file->size);if(file->file_handle)close((int)(intptr_t)file->file_handle-1);
#endif
    memset(file,0,sizeof(*file));
}
