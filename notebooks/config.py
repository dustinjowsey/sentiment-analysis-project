import sys
import os

#append the root folder to path
root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if root not in sys.path:
    sys.path.append(root)
    
#Append src to path so we can use the modules in src in our notebooks easier
#Just include config in your notebook
src = os.path.abspath(os.path.join(root, 'src'))
if src not in sys.path:
    sys.path.append(src)