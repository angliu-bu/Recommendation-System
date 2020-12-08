# main.py
# simulate different request coming into the system

from Webserver import WebServer, Request, Action

configMap = {"numberToServe": 10, "data_dir": "DATA"}
server = WebServer(configMap) # create a web server
server.start() # load all the data in the database, start the first model training

# now experiment
reqX1 = Request(userId='X1') # an anonymous user, use string.
req1 = Request(userId=1) # if it is a registered user, we use integer
print(reqX1)
print(req1)

# return recommendations
recX1 = server.renderRecommendation(reqX1)
print(recX1)

#
rec1 = server.renderRecommendation(req1)
print(rec1)

# user 1 give rating of 5 for item 255
action1 = Action(1, 255, 5)
print (server.getFromInventory(255))
# server take action to update for user 1 using online model
server.getAction(action1)
rec1_afteraction = server.renderRecommendation(req1)
print(rec1_afteraction)

# anonymous user give rating of 5 for item 123
actionX1 = Action('X1', 123, 5)
print (server.getFromInventory(123))
server.getAction(actionX1)
recX1_afteraction = server.renderRecommendation(reqX1)
print(recX1_afteraction)

# update offline models
server.increment()
recX1_aftercleaning = server.renderRecommendation(reqX1)
print(recX1_aftercleaning)


req19 = Request(userId=19) # the one with very few history
rec19 = server.renderRecommendation(req19)
print(rec19)
