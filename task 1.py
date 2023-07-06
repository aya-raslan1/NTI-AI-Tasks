#task 1
#get user email that ends with@NTI.com and get pw that matchs,if it's wrong continue else welcome
while True:
   new_email=input("plz enter new email: ")
   if (new_email.endswith("@nti.com")):
       break
   else:
      print("plz enter email end by @nti.com")
stored_password=input("plz enter new pssword: ")
while True:
   email=input("plz enter your email: ")
   password=input("plz enter your pssword: ")
   if (email==new_email) and (stored_password==password):
       print("WELCOME")
       break
   else:
      print("plz enter true information")