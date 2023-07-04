stored_password="555"
while len(stored_password)>0:
   email=input("plz enter your email: ")
   password=input("plz enter your pssword: ")
   if (email.endswith("@nti.com")) and (stored_password==password):
       print("WELCOME")
       break
   else:
      print("plz enter true information")

