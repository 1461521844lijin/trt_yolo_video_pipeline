/* libtiff/tif_config.h.cmake.in.  Not generated, but originated from autoheader.  */
/* This file must be kept up-to-date with needed substitutions from libtiff/tif_config.h.in. */

/* Support CCITT Group 3 & 4 algorithms */
#define CCITT_SUPPORT 1

/* Pick up YCbCr subsampling info from the JPEG data stream to support files
   lacking the tag (default enabled). */
#define CHECK_JPEG_YCBCR_SUBSAMPLING 1

/* enable partial strip reading for large strips (experimental) */
/* #undef CHUNKY_STRIP_READ_SUPPORT */

/* Support C++ stream API (requires C++ compiler) */
/* #undef CXX_SUPPORT */

/* enable deferred strip/tile offset/size loading (experimental) */
/* #undef DEFER_STRILE_LOAD */

/* Define to 1 if you have the <assert.h> header file. */
#define HAVE_ASSERT_H 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the <fcntl.h> header file. */
#define HAVE_FCNTL_H 1

/* Define to 1 if you have the `getopt' function. */
#define HAVE_GETOPT 1

/* Define to 1 if you have the <GLUT/glut.h> header file. */
/* #undef HAVE_GLUT_GLUT_H */

/* Define to 1 if you have the <GL/glut.h> header file. */
/* #undef HAVE_GL_GLUT_H */

/* Define to 1 if you have the <GL/glu.h> header file. */
/* #undef HAVE_GL_GLU_H */

/* Define to 1 if you have the <GL/gl.h> header file. */
/* #undef HAVE_GL_GL_H */

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <io.h> header file. */
/* #undef HAVE_IO_H */

/* Define to 1 if you have the `jbg_newlen' function. */
/* #undef HAVE_JBG_NEWLEN */

/* Define to 1 if you have the `lfind' function. */
#define HAVE_LFIND 1

/* Define to 1 if you have the `mmap' function. */
#define HAVE_MMAP 1

/* Define to 1 if you have the <OpenGL/glu.h> header file. */
/* #undef HAVE_OPENGL_GLU_H */

/* Define to 1 if you have the <OpenGL/gl.h> header file. */
/* #undef HAVE_OPENGL_GL_H */

/* Define to 1 if you have the <search.h> header file. */
#define HAVE_SEARCH_H 1

/* Define to 1 if you have the `setmode' function. */
/* #undef HAVE_SETMODE */

/* Define to 1 if you have the `snprintf' function. */
#define HAVE_SNPRINTF 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the `strcasecmp' function. */
#define HAVE_STRCASECMP 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strtol' function. */
#define HAVE_STRTOL 1

/* Define to 1 if you have the `strtoll' function. */
/* #undef HAVE_STRTOLL */

/* Define to 1 if you have the `strtoul' function. */
#define HAVE_STRTOUL 1

/* Define to 1 if you have the `strtoull' function. */
#define HAVE_STRTOULL 1

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* 8/12 bit libjpeg dual mode enabled */
/* #undef JPEG_DUAL_MODE_8_12 */

/* 12bit libjpeg primary include file with path */
#define LIBJPEG_12_PATH 

/* Support LZMA2 compression */
/* #undef LZMA_SUPPORT */

/* Support ZSTD compression */
/* #undef ZSTD_SUPPORT */

/* Support WEBP compression */
/* #undef WEBP_SUPPORT */

/* Name of package */
#define PACKAGE ""

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT ""

/* Define to the full name of this package. */
#define PACKAGE_NAME ""

/* Define to the full name and version of this package. */
#define PACKAGE_STRING ""

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME ""

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION ""

/* The size of `signed int', as computed by sizeof. */
#define SIZEOF_SIGNED_INT 4

/* The size of `signed long', as computed by sizeof. */
#define SIZEOF_SIGNED_LONG 8

/* The size of `signed long long', as computed by sizeof. */
#define SIZEOF_SIGNED_LONG_LONG 8

/* The size of `unsigned char *', as computed by sizeof. */
#define SIZEOF_UNSIGNED_CHAR_P 8

/* The size of `unsigned int', as computed by sizeof. */
#define SIZEOF_UNSIGNED_INT 4

/* The size of `unsigned long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG 8

/* The size of `unsigned long long', as computed by sizeof. */
#define SIZEOF_UNSIGNED_LONG_LONG 8

/* The size of `unsigned short', as computed by sizeof. */
#define SIZEOF_UNSIGNED_SHORT 2

/* Default size of the strip in bytes (when strip chopping enabled) */
#define STRIP_SIZE_DEFAULT 8192

/* Signed 32-bit type formatter */
#define TIFF_INT32_FORMAT "%d"

/* Signed 64-bit type formatter */
#define TIFF_INT64_FORMAT "%ld"

/* Pointer difference type formatter */
#define TIFF_PTRDIFF_FORMAT "%ld"

/* Unsigned size type formatter */
#define TIFF_SIZE_FORMAT "%lu"

/* Signed size type formatter */
#define TIFF_SSIZE_FORMAT "%ld"

/* Unsigned 32-bit type formatter */
#define TIFF_UINT32_FORMAT "%u"

/* Unsigned 64-bit type formatter */
#define TIFF_UINT64_FORMAT "%lu"

/* Unsigned 8-bit type */
#define TIFF_UINT8_T unsigned char

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#undef TIME_WITH_SYS_TIME

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
/* #undef TM_IN_SYS_TIME */

/* define to use win32 IO system */
/* #undef USE_WIN32_FILEIO */

/* Version number of package */
#define VERSION ""

/* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
   significant byte first (like Motorola and SPARC, unlike Intel). */
#if defined AC_APPLE_UNIVERSAL_BUILD
# if defined __BIG_ENDIAN__
#  define WORDS_BIGENDIAN 1
# endif
#else
# ifndef WORDS_BIGENDIAN
#  undef WORDS_BIGENDIAN
# endif
#endif

/* Number of bits in a file offset, on hosts where this is settable. */
#define _FILE_OFFSET_BITS 64

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
#define inline inline
#endif

/* Define to `long int' if <sys/types.h> does not define. */
#undef off_t

/* Define to `unsigned int' if <sys/types.h> does not define. */
#undef size_t
