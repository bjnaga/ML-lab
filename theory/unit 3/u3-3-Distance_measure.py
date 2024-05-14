# Euclidean Distance
from scipy.spatial import distance
x = [3,6,9]
y = [1,0,1]

print("Euclidean Distance ",distance.euclidean(x,y))

# Manhattan distancing/ taxicab distance / cityblock distance
print("Manhattan distancing ",distance.cityblock(x,y))


# Minkowski Distance 

print("Minkowski Distance ",distance.minkowski(x,y,p=3))
