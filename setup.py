import setuptools
import shrub

setuptools.setup(
    name=shrub.NAME,
    version=shrub.VERSION,
    description=shrub.DESCRIPTION,
    packages=setuptools.find_packages(),
    python_requires='>=3.5.*, <4',

    author='王振华（Zhenhua WANG）',
    author_email='i@jackwish.net',
    url="https://jackwish.net/shrub",

    project_urls={
        'Bug Reports': 'https://github.com/jackwish/shrub/issues',
        'Source': 'https://github.com/jackwish/shrub',
    },
)
