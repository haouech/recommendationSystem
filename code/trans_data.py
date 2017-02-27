import sys

f = open("../data/users.dat", 'r')
file = f.read()
file = file.replace("::", ",")
fout = open("../data/users.csv", 'w')

fout.write(file)
fout.close()

f = open("../data/movies.dat", 'r')
file = f.read()

file = file.replace(",", " ")
file = file.replace("::", ",")
fout = open("../data/movies.csv", 'w')

fout.write(file)
fout.close()

f = open("../data/ratings.dat", 'r')
file = f.read()

file = file.replace("::", ",")
fout = open("../data/ratings.csv", 'w')

fout.write(file)
fout.close()
