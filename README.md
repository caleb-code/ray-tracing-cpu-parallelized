# Parallelized Ray Tracing (Spheres)
```python
import numpy as np
import time
import matplotlib.pyplot as plt
import sympy as sp
import random
```
Imports
```python
x, y, z, a, b, c, r, k, h, l, m, q, t, j,d  = sp.symbols("x y z a b c r k h l m q t j d")
plane = sp.Eq(r**2, (x-h)**2 + (y-k)**2 + (z-j)**2)
x = a*t+l
y = b*t+m
z = c*t+q
soln_alg = sp.solve(plane.subs(x,x).subs(y,y).subs(z,z), t)
```
Solve for the distance $t$ at intersection for any ray defined by the equations $x=at+l$, $y=bt+m$, and $z=ct+q$ and sphere defined by $(x-h)^2 + (y-k)^2 + (z-j)^2=r^2$
```python
def safe_sqrt(x):
  safe_inds = np.where(x >= 0, x, np.nan)
  return np.sqrt(safe_inds)
```
Safe square root of any number x. When x is negative set x to zero.\
$$safesqrt(x) = sqrt(max(0, x))$$
```python
class Sphere:
  def __init__(self,r,h,k,j,color,emission,reflection):
    self.radius = r
    self.coords = [h,k,j]
    self.emission = emission
    self.color = color
    self.reflection = reflection
  def pos(self):
    return self.coords, self.radius
  def line_instersect_sphere(self, a,b,c,r,h,k,j,l,m,q):
    out = np.zeros(b.shape)
    discriminant = safe_sqrt(-a**2*j**2 + 2*a**2*j*q - a**2*k**2 + 2*a**2*k*m - a**2*m**2 - a**2*q**2 + a**2*r**2 + 2*a*b*h*k - 2*a*b*h*m - 2*a*b*k*l + 2*a*b*l*m + 2*a*c*h*j - 2*a*c*h*q - 2*a*c*j*l + 2*a*c*l*q - b**2*h**2 + 2*b**2*h*l - b**2*j**2 + 2*b**2*j*q - b**2*l**2 - b**2*q**2 + b**2*r**2 + 2*b*c*j*k - 2*b*c*j*m - 2*b*c*k*q + 2*b*c*m*q - c**2*h**2 + 2*c**2*h*l - c**2*k**2 + 2*c**2*k*m - c**2*l**2 - c**2*m**2 + c**2*r**2)
    out = np.where(np.isnan(discriminant), out, np.inf)
    negative_root = ((a*h - a*l + b*k - b*m + c*j - c*q - discriminant)/(a**2 + b**2 + c**2))
    positive_root = ((a*h - a*l + b*k - b*m + c*j - c*q + discriminant)/(a**2 + b**2 + c**2))
    out = np.minimum(np.where(negative_root<=.1, np.inf, negative_root),
           np.where(positive_root<=.1, np.inf, negative_root))
    return out
  def intersect_ray(self, a, bc, lmq): # a,b,c = line slope; r = sphere radius; h,k,j = position of sphere; lmq = position of line
    b = bc[:,0]
    c = bc[:,1]
    r = self.radius
    h,k,j = self.coords
    l,m,q = lmq
    ans = self.line_instersect_sphere(a,b,c,r,h,k,j,l,m,q)
    return ans
```
Defines a sphere with position, radius, color, emission, and reflection. The `intersect_ray` method returns the distance to intersection of a ray with the sphere.
```python
scene = []
scene.append(Sphere(1,0,0,5,[255,0,0],0,.3))
scene.append(Sphere(1,2,0,5,[0,255,0],0,.3))
scene.append(Sphere(1,4,0,5,[0,0,255],0,.3))
scene.append(Sphere(1,0,2,5,[255,255,0],0,.3))
scene.append(Sphere(1,2,2,5,[255,0,255],0,.3))
scene.append(Sphere(1,4,2,5,[0,255,255],0,.3))
scene.append(Sphere(10000,0,-10004,5,[255,255,255],0,.3))
scene.append(Sphere(.5,1,1,2,[255,255,255],10,0))
```
Define the scene with spheres. The last sphere is a light source.
```python
width = 500
height = 500
N = width*height
sq_width = int(np.sqrt(N))
fov = np.pi/4
pixel_width = 2*np.tan(fov/2)/width
image = np.zeros((width, height, 3))
```
Define image dimensions and field of view.
```python
import multiprocessing
def compute_subset(start, end):
  subset_image = np.zeros((end-start, width, 3))
  for i in range(start, end):
    for j in range(width):
      x_dir = (j - width/2)*pixel_width
      y_dir = (i - height/2)*pixel_width
      direction = np.array([x_dir, y_dir, 1])
      direction = direction/np.linalg.norm(direction)
      origin = np.array([1,1,0])
      color = np.array([0,0,0])
      reflection = 1
      for bounce in range(5):
        min_dist = np.inf
        hit_sphere = None
        for sphere in scene:
          dist = sphere.intersect_ray(direction[0], direction[1:], origin)
          if dist < min_dist:
            min_dist = dist
            hit_sphere = sphere
        if hit_sphere is None:
          break
        hit_point = origin + direction*min_dist
        normal = (hit_point - np.array(hit_sphere.coords))/hit_sphere.radius
        color += reflection*np.array(hit_sphere.color)*hit_sphere.emission
        reflection *= hit_sphere.reflection
        if reflection < 0.01:
          break
        direction = direction - 2*np.dot(direction, normal)*normal
        origin = hit_point + normal*0.001
      subset_image[i-start, j] = color
  return subset_image
```
Compute a subset of the image. This function will be called in parallel.
```python
if __name__ == '__main__':
  num_processes = multiprocessing.cpu_count()
  pool = multiprocessing.Pool(processes=num_processes)
  step = height//num_processes
  results = []
  for i in range(num_processes):
    start = i*step
    end = (i+1)*step if i < num_processes-1 else height
    results.append(pool.apply_async(compute_subset, (start, end)))
  pool.close()
  pool.join()
  for i, result in enumerate(results):
    start = i*step
    end = (i+1)*step if i < num_processes-1 else height
    image[start:end] = result.get()
  plt.imshow(image.astype(int))
  plt.show()
```
Parallelize the computation of the image using multiprocessing.
