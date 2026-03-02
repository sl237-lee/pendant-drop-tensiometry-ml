import fileinput
import sys

# Fix the file
with open('src/physics/young_laplace.py', 'r') as f:
    content = f.read()

content = content.replace('np.trapz', 'np.trapezoid')

with open('src/physics/young_laplace.py', 'w') as f:
    f.write(content)

print("✅ Fixed numpy.trapz → numpy.trapezoid")
