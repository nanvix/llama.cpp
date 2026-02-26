/*
 * nanvix_stubs.c - Missing POSIX stubs for Nanvix
 *
 * Copyright(c) The Maintainers of Nanvix.
 * Licensed under the MIT License.
 *
 * Provides stub implementations for POSIX functions that libc++ and
 * llama.cpp reference but are not available on Nanvix.
 */

#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

/* DSO handle for static builds (no dynamic shared objects on Nanvix) */
void *__dso_handle = (void *)0;

/*
 * C++ global constructors/destructors support.
 *
 * The Nanvix linker script (user.ld) does not include .init_array/.fini_array
 * sections, so we force them to be retained and call constructors from _init().
 * The _init() function is called by the Nanvix C runtime before main().
 */

/* Symbols marking the constructor/destructor arrays (provided by the linker) */
typedef void (*init_fn)(void);
extern init_fn __init_array_start[] __attribute__((weak));
extern init_fn __init_array_end[]   __attribute__((weak));

void _init(void) {
    /* Call all global constructors registered in .init_array */
    if (__init_array_start && __init_array_end) {
        for (init_fn *fn = __init_array_start; fn < __init_array_end; fn++) {
            if (*fn) {
                (*fn)();
            }
        }
    }
}

void _fini(void) {
    /* Destructors are not needed for our use case */
}

/* pthread_mutexattr_settype - stub (Nanvix only has basic mutexes) */
int pthread_mutexattr_settype(void *attr, int type) {
    (void)attr;
    (void)type;
    return 0;
}

/* pathconf - return reasonable defaults for Nanvix */
long pathconf(const char *path, int name) {
    (void)path;
    switch (name) {
#ifdef _PC_PATH_MAX
        case _PC_PATH_MAX: return 4096;
#endif
#ifdef _PC_NAME_MAX
        case _PC_NAME_MAX: return 255;
#endif
        default: return 256;
    }
}

/* getcwd stub — return "/" as current directory if real getcwd fails */
char *__wrap_getcwd(char *buf, size_t size) __attribute__((weak));
char *__wrap_getcwd(char *buf, size_t size) {
    if (buf && size >= 2) {
        buf[0] = '/';
        buf[1] = '\0';
        return buf;
    }
    errno = ERANGE;
    return NULL;
}

/* statvfs - stub (no filesystem stats on Nanvix) */
int statvfs(const char *path, void *buf) {
    (void)path;
    (void)buf;
    errno = ENOSYS;
    return -1;
}

/* openat - stub (no directory-relative file ops on Nanvix) */
int openat(int dirfd, const char *pathname, int flags, ...) {
    (void)dirfd;
    (void)pathname;
    (void)flags;
    errno = ENOSYS;
    return -1;
}

/* fdopendir - stub (no directory streams from fd on Nanvix) */
void *fdopendir(int fd) {
    (void)fd;
    errno = ENOSYS;
    return NULL;
}

/*
 * ggml_fopen — Override to use larger stdio buffers on Nanvix.
 *
 * newlib's default BUFSIZ is 1024 bytes. When loading a 462 MB model,
 * fread() calls read() ~462K times with 1KB chunks, creating enormous
 * per-call overhead through the POSIX layer. Setting a 256 KB buffer
 * reduces the call count to ~1800 and nearly eliminates this overhead.
 *
 * Linked via -Wl,--wrap=ggml_fopen so our version replaces the original.
 */
extern FILE *__real_ggml_fopen(const char *fname, const char *mode);

FILE *__wrap_ggml_fopen(const char *fname, const char *mode) {
    FILE *f = __real_ggml_fopen(fname, mode);
    if (f) {
        setvbuf(f, NULL, _IOFBF, 256 * 1024);
    }
    return f;
}
