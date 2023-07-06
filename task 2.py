#task 2
#get user email that ends with ØNTI com and get pw that matchs if it's wrong continue
#also use Inheritance class
class acc:
    def __init__(self,email,password):
        self.email=email
        self.password=password
class login (acc):
    def __init__ (self,email,password):
        super().__init__(email,password)
while True:
   new_email=input("plz enter new email: ")
   if (new_email.endswith("@nti.com")):
       break
   else:
      print("plz enter email end by @nti.com")
new_password=input("plz enter new pssword: ")
user1=login(new_email,new_password)

while True:
   email=input("plz enter your email: ")
   password=input("plz enter your pssword: ")
   if (email.endswith("@nti.com")) and (user1.password==password):
       print("WELCOME")
       break
   else:
      print("plz enter true information")