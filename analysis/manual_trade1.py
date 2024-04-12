import numpy as np
import matplotlib.pyplot as plt

# Calculate the utility function U(L, H) for the given values of L and H bases on a continous price function (not accurate)
def U(L, H, b):
    return (1000-L)*(0.5*(L**2-900**2) - 900*b*(L-900)) + (1000-H)*(0.5*(H**2-L**2) - 900*b*(H-L))

# Calculate the utility function U(L, H) for the given values of L and H bases on a discrete price function (accurate)
def U_disc(L, H, a, b):
    return (1000-L)*(0.5*(a*(L-900)*(L+900-1)) +(L-900)*(b-a*900)) + (1000-H)*(0.5*a*((H-L)*(H+L-1))+(H-L)*(b-a*900))*np.where(H>L, 1, 0)

# Calculate the utility function U(L, H) for the given values of L and H bases on a discrete price function by manually looping through the values
def U_basic(l, h, a, b):
    res = np.zeros((l.shape[0], h.shape[0]))
    for i, l_i in enumerate(l):
        k1 = np.linspace(900, l_i-1, int(l_i-900))
        for j, h_j in enumerate(h):
            res[i, j] += (1000-l_i)*np.sum(a*k1 -a*900 + b)
            if h_j > l_i:
                k2 = np.linspace(l_i, h_j-1, int(h_j-l_i))
                res[i, j] += (1000-h_j)*np.sum(a*k2 -a*900 + b)
    return res

#the values of L and H to search
l = np.linspace(900, 1000, 101)
h = np.linspace(900, 1000, 101)

b = 1

#print the values of l-900 + 1 
print((l+1 - 900)+b )

#print the sum of l-900 + 1, alpha should be 1/np.sum((l+1 - 900) )
print(f"a = 1/{np.sum((l+1 - 900+b))}" )

a = 1/np.sum((l+1 - 900)+b)

#make meshed values based on l and h
L, H = np.meshgrid(l, h, indexing='ij')

#calculate the utility function U(L, H) for the given values of L and H
u_d = U_disc(L, H, a, a+b*a)

#plot the utility function U(L, H) for the given values of L and H
fig, ax = plt.subplots()
ax.imshow(U_disc(L, H, a, a+b*a), origin='lower')
ax.set_title('U(L, H) calculated vectorized')
ax.set_xlabel('H')
ax.set_ylabel('L')

#calculate the utility function U(L, H) for the given values of L and H using the loop approach to verify the vectorized approach
u_b = U_basic(l, h, 1/5151, 1/5151)

#compare the two approaches
#print(u_d)
#print(u_b)

#calculate the norm of the difference between the two approaches. This should be a small number
print(np.linalg.norm(u_d-u_b))

#find the maximum value of the utility function U(L, H) and the corresponding values of L and H
u = u_d
l_max, h_max = np.unravel_index(np.argmax(u), u.shape)
print(l[l_max], h[h_max])

#plot the utility function U(L, H) for the given values of L and H using the loop approach
fig, ax = plt.subplots()
ax.pcolormesh(h, l, u)
ax.contour(h, l, u, levels=10)
ax.scatter(h[h_max], l[l_max], color='red', label=f'max at L={l[l_max]}, H={h[h_max]}')
ax.legend()
ax.set_title('U(L, H) calculated loop wise')
ax.set_xlabel('H')
ax.set_ylabel('L')
plt.show()