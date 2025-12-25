guest_list = ['cristiano ronaldo', 'neymar jr', 'lionel messi']

message = f"Mr. {guest_list[0].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[1].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[2].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)

print(f"Unfortunately, Mr. {guest_list[2].title()} will not be able to make it.")
guest_list[2] = 'marcelo'
print(guest_list)

message = f"Mr. {guest_list[0].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[1].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[2].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)

guest_list.insert(0, 'pepe')
guest_list.insert(3, 'sergio ramos')
guest_list.append('toni kroos')
print(guest_list)

message = f"Mr. {guest_list[0].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[1].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[2].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[3].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[4].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[5].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)

print(f"Unfortunately, due to unforseen limitations, I will now only be able to accommodate 2 individuals for this dinner.")
guest_list.pop(0)
print(guest_list)
print(f"Unfortunately, due to unforseen limitations, I regret to inform you with apologies that I can no longer accommodate you at this dinner.")
guest_list.pop(2)
print(guest_list)
print(f"Unfortunately, due to unforseen limitations, I regret to inform you with apologies that I can no longer accommodate you at this dinner.")
guest_list.pop(2)
print(guest_list)
print(f"Unfortunately, due to unforseen limitations, I regret to inform you with apologies that I can no longer accommodate you at this dinner.")
guest_list.pop(2)
print(f"Unfortunately, due to unforseen limitations, I regret to inform you with apologies that I can no longer accommodate you at this dinner.")
print(guest_list)

message = f"Mr. {guest_list[0].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)
message = f"Mr. {guest_list[1].title()}, you are hereby cordially invited to a footballing dinner for the ages!"
print(message)

del guest_list[0]
del guest_list[0]
print(guest_list)