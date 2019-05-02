import itertools
import pickle
import numpy as np
import random
import sys
import igraph as ig
from igraph import plot

if len(sys.argv) > 1:
	COUNT = int(sys.argv[1])
	MIN_EDGE_VALUE = int(sys.argv[2])
	LOG_FN = sys.argv[3]
else:
	COUNT          = 10000 
	MIN_EDGE_VALUE = 4
	LOG_FN = "cc_summary_res"

EDGELIST_FN = "pol_300_year_00_50_new_weighted"

IGNORE_SUBS_LISTS = [
	    # ["politics", "AskReddit", "worldnews", "news", "funny", "pics"]]	
	    ["politics"],
	    ["politics", "AskReddit", "worldnews", "news", "funny", "pics"],
		["politics", "AskReddit", "worldnews", "news", "funny", "pics", "todayilearned", "gaming", "aww", "videos", "movies"],
		["politics", "AskReddit", "worldnews", "news", "funny", "pics", "todayilearned", "gaming", "aww", "videos", "movies", "gifs", "PoliticalHumor", "Showerthoughts", "interestingasfuck", "WTF"],
		["politics", "AskReddit", "worldnews", "news", "funny", "pics", "todayilearned", "gaming", "aww", "videos", "movies", "gifs", "PoliticalHumor", "Showerthoughts", "interestingasfuck", "WTF", "mildlyinteresting", "trashy", "unpopularopinion", "BlackPeopleTwitter", "technology"],
		["politics", "AskReddit", "worldnews", "news", "funny", "pics", "todayilearned", "gaming", "aww", "videos", "movies", "gifs", "PoliticalHumor", "Showerthoughts", "interestingasfuck", "WTF", "mildlyinteresting", "trashy", "unpopularopinion", "BlackPeopleTwitter", "technology", "science", "PublicFreakout", "OldSchoolCool", "AmItheAsshole", "nottheonion", "relationship_advice", "MurderedByWords", "RoastMe", "WhitePeopleTwitter", "oddlysatisfying", "atheism", "AdviceAnimals", "insanepeoplefacebook", "memes", "television", "nba", "AskMen", "nfl", "Futurology", "Damnthatsinteresting"]]

IGNORE_SUBS = IGNORE_SUBS_LISTS[3]

def testplot(graph, lo, filename="social_network.png"):
    # visual_style = dict()
    # visual_style["vertex_color"] = "blue"
    # visual_style["edge_width"] = 1
    # visual_style["layout"] = lo
    # visual_style["bbox"] = (1200, 1000)
    # visual_style["margin"] = 100
    plot(graph, filename, layout=lo, vertex_size=4, edge_width=0, bbox=(600,1200)) 

def plog(str, fo):
	print(str)
	fo.write(str+"\n")

log_file = open("results/{}.txt".format(LOG_FN), "w")
edge_list = pickle.load(open("data/{}.p".format(EDGELIST_FN), "rb"))
plog("opened edgelist from: data/{}.p".format(EDGELIST_FN), log_file)
plog("random draw of {}, threshold {}".format(COUNT, MIN_EDGE_VALUE), log_file)

el = random.sample(edge_list, COUNT)

#### Building The Projection Graph
### These steps are the same for each iteration
user_map = dict()
subs_map = dict()
uid_uname_map = []
sid_sname_map = []

# First pass builds the user map
u_id = 0
s_id = 0
for user in el:
	un = user[0]
	user_map[un] = u_id
	u_id += 1
	uid_uname_map.append(un)

# Second pass builds the sub map
s_id = 0
for user in el:
	un = user[0]
	subs = user[1]
	for sub_name in subs:
		if sub_name not in subs_map:
			subs_map[sub_name] = s_id
			s_id += 1
			sid_sname_map.append(sub_name)

# Bi-Adj Matrix 
B = np.zeros((len(user_map), len(subs_map)), dtype=np.int16)

i = 0
for user in el:
	un = user[0]
	user_id = user_map[un]
	subs = user[1]
	for sub_name in subs:
		sub_id = subs_map[sub_name]
		if sub_name not in IGNORE_SUBS:
			B[user_id][sub_id] = 1.0
	i += 1
	if i >= COUNT:
		break

u, s, vh = np.linalg.svd(B, full_matrices=True)

u_t2 = u[:,:2] # Top two columns of U - Embed the users
s_t2 = s[:2]   # Top two singular values of A
v_t2 = vh[:2]   # Top two rows of vH   - Embed the subs

s_t2_sqr = np.zeros((2,2))
s_t2_sqr[0,0] = s_t2[0]
s_t2_sqr[1,1] = s_t2[1]
s_t2 = s_t2_sqr

### experiment
	# # To embed subs, we do Vsub = sub' x u2 x s2^-1
	# # So lets save this right multiplier
	# sub_embed = np.matmul(u_t2, np.linalg.inv(s_t2))

	# # To test, lets embed the first sub (column) of B
	# first_col = B[:,:1]
	# print(np.matmul(np.transpose(first_col), sub_embed))

	# # To embed users, we do Uuser = user x (v2^-1)' x s2^-1
	# # So lets save the right multiplier
	# user_embed = np.matmul(np.transpose(np.linalg.inv(v_t2)), np.linalg.inv(s_t2))

### Embedding?
# The L/R matrices above are the embeddings for our observed nodes

### Plotting?
# Need an igraph, along with a layout
num_users = B.shape[0]
num_subs  = B.shape[1]
zero_u = np.zeros((num_users, num_users))
zero_s = np.zeros((num_subs, num_subs))

A = np.block([[zero_u, B], [np.transpose(B), zero_s]])
layout = np.block([[u_t2], [np.transpose(v_t2)]])

full_graph = ig.Graph.Adjacency(A.tolist(), mode=ig.ADJ_MAX)

# Adding attributes to the nodes. 
for i in range(0, num_users):
	full_graph.vs[i]["color"] = "blue"

for i in range(num_users, num_subs+num_users-1):
	full_graph.vs[i]["color"] = "red"


g_layout = ig.Layout(layout.tolist())

testplot(full_graph, g_layout, filename="test.pdf")
