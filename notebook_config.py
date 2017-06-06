# Configuration file for ipython.

c = get_config()

# Kernel config
c.IPKernelApp.pylab = 'inline'

# Notebook config
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.open_browser = False
c.NotebookApp.password = u'sha1:d707be900181:3872150813e63c5d7d6bdbb76407b09fdefd8d53' #udacity2017
c.NotebookApp.port = 8889
c.NotebookApp.base_url = '/ipython/'
c.NotebookApp.trust_xheaders = True
c.NotebookApp.tornado_settings = {'static_url_prefix': '/ipython/static/'}
c.NotebookApp.allow_origin = '*'
