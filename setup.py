from setuptools import setup

def readme():
  with open('README.md') as f:
    README = f.read()
    return README

setup(
  name = 'cvxmg',         
  version = '2.0.0',      
  description = 'Planning of Microgrids considering Demand Side Management Strategies using Disciplined Convex Deterministic and StochasticProgramming',   
  long_description = readme(),
  long_description_content_type='text/markdown',
  url = 'https://github.com/juancaoviedo/cvxmg',      
  author = 'Juan Carlos Oviedo Cepeda',                 
  author_email = 'juan.oviedo@correo.uis.edu.co',      
  license='MIT',        
  packages = ['cvxmg'],
  include_package_data=True,   
  download_url = 'https://github.com/juancaoviedo/cvxmg.git',    
  keywords = ['convex', 'microgrids', 'optimization', 'stochastic', 'demand-side management'],       
  install_requires=[            
          'numpy',
          'pandas',
          'cvxpy',
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',            
    'Intended Audience :: Developers',            
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',     
    'Programming Language :: Python :: 3.6',      
    'Programming Language :: Python :: 3.7',
  ],
)
