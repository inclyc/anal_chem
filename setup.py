import os

from setuptools import setup, find_packages


def _process_requirements():
    packages = open('requirements.txt').read().strip().split('\n')
    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip install {}'.format(pkg))
            assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)
    return requires


setup(
    name='anal-chem',
    version='0.0.0',
    url='https://github.com/Enzyme125/anal_chem',
    license='MIT',
    author='Y.C. Long',
    author_email='847072154@qq.com',
    install_requires=_process_requirements(),
    description='分析化学中常用函数实现',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese (Simplified)',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Utilities',
    ],
    package_dir={'anal_chem': 'anal_chem'},
    packages=find_packages('.'),
    python_requires='>=3.8, <4',
    keywords=['chemistry','analytical chemistry']
)

