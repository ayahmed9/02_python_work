motorcycles = ['honda', 'yamaha', 'suzuki']
#print(motorcycles)


#motorcycles[0] = 'ducati'

motorcycles.append('ducati')

#motorcycles = []

#motorcycles.append('honda')
#motorcycles.append('yamaha')
#motorcycles.append('suzuki')

#motorcycles.insert(0, 'ducati')

#del motorcycles[1]

#popped_motorcycle = motorcycles.pop()
print(motorcycles)
#print(popped_motorcycle)

#last_owned = motorcycles.pop()
#print(f"The last motorcycle I owned was a {last_owned.title()}.")

#first_owned = motorcycles.pop(0)
#print(f"The first motorcycle I owned was a {first_owned.title()}.")

#motorcycles.remove('ducati')

too_expensive = 'ducati'
motorcycles.remove(too_expensive)
print(motorcycles)
print(f"\nA {too_expensive.title()} is too expensive for me.")

