import sys
import random
import Queue

if len(sys.argv) < 2:
    print 'usage: ', sys.argv[0], '[file]'
    sys.exit(1)

data_file = open(sys.argv[1],'w')

width = 5
height = 5
colors = 3
classes = 10
samples_per_class = 10

data_size = width * height * colors

def generate_template():
    template = []
    for x in range(data_size):
        template.append(random.randint(0, 1)) 
    return template

output=''
space=' '
#generate data 
for i in range(classes):
    template = generate_template()
    template.append(i)
    for j in range(samples_per_class):
        #data_file.write(''.join(str(x)+' ' for x in template))
        #data_file.write('\n')
        output += space.join(str(x) for x in template)
        output +='\n'

shuffled = output.split('\n')
random.shuffle(shuffled)
#print shuffled
output = ''
i = 0
for x in shuffled:
    if x:
        output += x + '\n'
        #print 'empty:', x,'; iteration', i
    #else:
        #output += x + '\n'
    i += 1
output = output[:-1]
data_file.write(output)
