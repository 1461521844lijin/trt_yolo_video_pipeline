#!/usr/bin/env python
import sys, re, os, time
from string import Template
from parse_tree import ParseTree, todict, constants
from filters import *

updated_files = []

def update_file(fname, content):
    if fname in updated_files:
        print('ERROR(gen_matlab.py): attemption to write file multiple times: {}'.format(fname))
        return
    updated_files.append(fname)
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            old_content = f.read()
        if old_content == content:
            #print('Up-to-date: {}'.format(fname))
            return
        print('Updating: {}'.format(fname))
    else:
        print('Writing: {}'.format(fname))
    with open(fname, 'wb') as f:
        f.write(content)


class MatlabWrapperGenerator(object):
    """
    MatlabWrapperGenerator is a class for generating Matlab mex sources from
    a set of C++ headers. MatlabWrapperGenerator objects can be default
    constructed. Given an instance, the gen() method performs the translation.
    """

    def gen(self, module_roots, modules, extras, output_dir):
        """
        Generate a set of Matlab mex source files by parsing exported symbols
        in a set of C++ headers. The headers can be input in one (or both) of
        two methods:
        1. specify module_root and modules
           Given a path to the OpenCV module root and a list of module names,
           the headers to parse are implicitly constructed.
        2. specifiy header locations explicitly in extras
           Each element in the list of extras must be of the form:
           'namespace=/full/path/to/extra/header.hpp' where 'namespace' is
           the namespace in which the definitions should be added.
        The output_dir specifies the directory to write the generated sources
        to.
        """
        # dynamically import the parsers
        from jinja2 import Environment, FileSystemLoader
        import hdr_parser

        # parse each of the files and store in a dictionary
        # as a separate "namespace"
        parser = hdr_parser.CppHeaderParser()

        ns  = dict((key, []) for key in modules)
        path_template = Template('${module}/include/opencv2/${module}.hpp')

        for module in modules:
            for module_root in module_roots:
                # construct a header path from the module root and a path template
                header = os.path.join(module_root, path_template.substitute(module=module))
                if os.path.isfile(header):
                    break
            else:
                raise Exception('no header found for module %s!' % module)

            # parse the definitions
            ns[module] = parser.parse(header)

        for extra in extras:
            module = extra.split("=")[0]
            header = extra.split("=")[1]
            ns[module] = ns[module] + parser.parse(header) if module in ns else parser.parse(header)

        # cleanify the parser output
        parse_tree = ParseTree()
        parse_tree.build(ns)

        # setup the template engine
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        jtemplate    = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)

        # add the custom filters
        jtemplate.filters['formatMatlabConstant'] = formatMatlabConstant
        jtemplate.filters['convertibleToInt'] = convertibleToInt
        jtemplate.filters['toUpperCamelCase'] = toUpperCamelCase
        jtemplate.filters['toLowerCamelCase'] = toLowerCamelCase
        jtemplate.filters['toUnderCase'] = toUnderCase
        jtemplate.filters['matlabURL'] = matlabURL
        jtemplate.filters['stripTags'] = stripTags
        jtemplate.filters['filename'] = filename
        jtemplate.filters['comment']  = comment
        jtemplate.filters['inputs']   = inputs
        jtemplate.filters['ninputs'] = ninputs
        jtemplate.filters['outputs']  = outputs
        jtemplate.filters['noutputs'] = noutputs
        jtemplate.filters['qualify'] = qualify
        jtemplate.filters['slugify'] = slugify
        jtemplate.filters['only'] = only
        jtemplate.filters['void'] = void
        jtemplate.filters['not'] = flip

        # load the templates
        tfunction  = jtemplate.get_template('template_function_base.cpp')
        tclassm    = jtemplate.get_template('template_class_base.m')
        tclassc    = jtemplate.get_template('template_class_base.cpp')
        tconst     = jtemplate.get_template('template_map_base.m')

        # create the build directory
        output_source_dir  = output_dir+'/src'
        output_private_dir = output_source_dir+'/private'
        output_class_dir   = output_dir+'/+cv'
        output_map_dir     = output_dir+'/map'
        if not os.path.isdir(output_source_dir):
          os.makedirs(output_source_dir)
        if not os.path.isdir(output_private_dir):
          os.makedirs(output_private_dir)
        if not os.path.isdir(output_class_dir):
          os.makedirs(output_class_dir)
        if not os.path.isdir(output_map_dir):
          os.makedirs(output_map_dir)

        # populate templates
        for namespace in parse_tree.namespaces:
            # functions
            for method in namespace.methods:
                populated = tfunction.render(fun=method, time=time, includes=namespace.name)
                update_file(output_source_dir+'/'+method.name+'.cpp', populated.encode('utf-8'))
            # classes
            for clss in namespace.classes:
                # cpp converter
                populated = tclassc.render(clss=clss, time=time)
                update_file(output_private_dir+'/'+clss.name+'Bridge.cpp', populated.encode('utf-8'))
                # matlab classdef
                populated = tclassm.render(clss=clss, time=time)
                update_file(output_class_dir+'/'+clss.name+'.m', populated.encode('utf-8'))

        # create a global constants lookup table
        const = dict(constants(todict(parse_tree.namespaces)))
        populated = tconst.render(constants=const, time=time)
        update_file(output_dir+'/cv.m', populated.encode('utf-8'))


if __name__ == "__main__":
    """
    Usage: python gen_matlab.py
                                --hdrparser /path/to/hdr_parser/dir
                                --moduleroot [ /path/to/opencv/modules /path/to/opencv_contrib/modules etc ]
                                --modules [core imgproc objdetect etc]
                                --extra namespace=/path/to/extra/header.hpp
                                --outdir /path/to/output/generated/srcs

    gen_matlab.py is the main control script for generating matlab source
    files from given set of headers. Internally, gen_matlab:
      1. constructs the headers to parse from the module root and list of modules
      2. parses the headers using CppHeaderParser
      3. refactors the definitions using ParseTree
      4. populates the templates for classes, function, enums from the
         definitions

    gen_matlab.py requires the following inputs:
    --hdrparser    the path to the header parser directory
                   (opencv/modules/python/src2)
    --moduleroot   (optional) paths to the opencv directories containing the modules
    --modules      (optional - required if --moduleroot specified) the modules
                   to produce bindings for. The path to the include directories
                   as well as the namespaces are constructed from the modules
                   and the moduleroot
    --extra        extra headers explicitly defined to parse. This must be in
                   the format "namepsace=/path/to/extra/header.hpp". For example,
                   the core module requires the extra header:
                   "core=/opencv/modules/core/include/opencv2/core/core/base.hpp"
    --outdir       the output directory to put the generated matlab sources. In
                   the OpenCV build this is "${CMAKE_CURRENT_BUILD_DIR}/src"
    """

    # parse the input options
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--hdrparser')
    parser.add_argument('--moduleroot', nargs='*', default=[], required=False)
    parser.add_argument('--modules', nargs='*', default=[], required=False)
    parser.add_argument('--extra', nargs='*', default=[], required=False)
    parser.add_argument('--outdir')
    args = parser.parse_args()

    # add the hdr_parser module to the path
    sys.path.append(args.hdrparser)

    # create the generator
    mwg = MatlabWrapperGenerator()
    mwg.gen(args.moduleroot, args.modules, args.extra, args.outdir)
