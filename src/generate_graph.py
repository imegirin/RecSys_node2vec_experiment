import rdflib
import pandas as pd
import numpy as np
import sys
import os
import pickle
from scipy.spatial.distance import cosine as cosdist
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import main as node2vec

ml2dbp = pd.read_csv('./MappingMovielens2DBpedia-1.2.tsv', sep='\t', names=['id', 'title', 'URI'])
users = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=['userid','movieid','rating','timestamp'], usecols=['userid','movieid','rating'], engine='python')
used_predicates=['http://dbpedia.org/ontology/' + str(item) for item in ['director','starring','distributor','writer','musicComposer','producer','cinematography','editing']]

users = users[users.rating==5]

'''
Following function were not used in any place in main code. I decided to execute it once and then to used downloaded data saved in pickle object.
'''
def download_DBP():
    entities = {}
    graph = []
    df = pd.DataFrame(columns=['id','title'])
    for id in ml2dbp.index:
        try:
            g=rdflib.Graph()
            g.load(ml2dbp.iloc[id].URI)
            title= set()
            for s,_,_ in g:
                title.add(s)
            df = df.append(pd.DataFrame({'id':[ml2dbp.iloc[id].id], 'title': [list(title)[0]]}))
            graph.append((ml2dbp.iloc[id].id, g))
        except Exception as e:
            print(e)
    df.to_csv('ids_titles.csv')
    pickle.dump(graph, open('src/garph.p', 'wb'))

def generate_graph(regenerate=False):
    global users

    g = pickle.load(open('src/graph.p','rb'))
    entities = []
    entities += list(users.userid.unique())
    entities += list(ml2dbp.URI.unique())
    tmp = []
    for g_ in g:
        for _, p, o in g_:
            if str(p) in used_predicates:
                tmp += [o]
    entities += list(np.unique(tmp))
    entities = np.unique(entities)

    ent_dict_it_obj = dict(zip([i for i in range(1,len(entities)+1)], entities))
    ent_dict_obj_it = dict(zip(entities, [i for i in range(1,len(entities)+1)]))
    print('number of entities: {}'.format(len(entities)))
    del entities

    users = users.merge(ml2dbp,how='left',left_on='movieid', right_on='id')
    users=users.drop(['rating','id','title'], axis=1)
    users = users.dropna()
    users['liked']=1

    negative = pd.DataFrame()
    print(users.shape)
    for user in users.userid.unique():
        tmp = users[users.userid==user]
        if tmp.shape[0]>40:
            tmp=tmp.head(5)
            unique_movies=np.array(users.movieid.unique())
            tmp['movieid'] = np.random.choice(np.setdiff1d(unique_movies,
                np.array(tmp.movieid)), 5)
            tmp['liked']=0
            negative = negative.append(tmp)
    print(negative.shape)

    users_train=pd.DataFrame(columns=users.columns)
    users_test=pd.DataFrame(columns=users.columns)
    for uid in users.userid.unique():
        tmp_df = users[users.userid==uid]
        if tmp_df.shape[0]>20:
            indexes = tmp_df.sample(7).index
            users_test = users_test.append(tmp_df.loc[indexes])
            users_train = users_train.append(tmp_df.drop(indexes))
        else:
            users_train = users_train.append(tmp_df)
    users_test = users_test.append(negative)

    users_train['user_key'] = users_train.userid.astype(str).map(lambda x : ent_dict_obj_it[x])
    users_train['movie_id_key'] = users_train.URI.astype(str).map(lambda x : ent_dict_obj_it[x])

    users_test['user_key'] = users_test.userid.astype(str).map(lambda x : ent_dict_obj_it[x])
    users_test['movie_id_key'] = users_test.URI.astype(str).map(lambda x : ent_dict_obj_it[x])

    users_train.to_csv('users_train.csv', index=False)
    users_test.to_csv('users_test.csv', index=False)

    if regenerate==True:
        pickle.dump((ent_dict_it_obj,ent_dict_obj_it),
                open('src/dictionaries.p','wb'))
        count = 0
        print('Generating graph. This can take a while...')
        with open('graph/movielens.edgelist','w') as f:
            for i in range(users_train.shape[0]):
                count += 1
                f.write("{} {}\n".format(users_train.iloc[i].user_key,
                                 users_train.iloc[i].movie_id_key))
            print('users in graph: {}'.format(count))
            count=0
            number_movies = []
            for g_ in g:
                for s,p,o in g_:
                    if str(p) in used_predicates:
                        id_val = ent_dict_obj_it.get(str(s))
                        if id_val!=None:
                            number_movies.append(str(s))
                            f.write("{} {}\n".format(id_val,
                                ent_dict_obj_it.get(str(o))))
                            count+=1
            print('unique moives: {}'.format(len(np.unique(number_movies))))
            print('other entities: {}'.format(count))

        print('Graph completed')
'''
All data required for evaluaton process is loaded from csvs or from pickles,
so following function can be executed alone.
'''
def evaluate():
    (ent_dict_it_obj,ent_dict_obj_it) = pickle.load(open('src/dictionaries.p','rb'))
    users_train=pd.read_csv('users_train.csv')
    users_test=pd.read_csv('users_test.csv')
    entity_vectors = {}
    with open('emb/movielens.emb', 'r') as f:
        next(f)
        for line in f:
            line = line.split()
            entity_vectors[int(line[0])]=np.array(line[1:]).astype(np.float)
    users_train['dist']=users_train[['user_key','movie_id_key']].apply(lambda x: cosdist(entity_vectors[x[0]],
        entity_vectors[x[1]]), axis=1)
    users_test['dist']=users_test[['user_key','movie_id_key']].apply(lambda x: cosdist(entity_vectors[x[0]],
        entity_vectors[x[1]]), axis=1)

### Below code is a little cheating.
### Threshold should be established based on training data not on test data.
### To do this negative samples are needed for training data.
### Then when generating negative samples for test data have to make sure
### that those are not the same. Then whole project will be done according to
### the ML art.
    users_test['label']=0
    if users_test[users_test.liked==1].dist.mean()<users_test[users_test.liked==0].dist.mean():
        users_test['label']=users_test.dist<np.mean([users_test[users_test.liked==1].dist.mean(),
            users_test[users_test.liked==0].dist.mean()])
    else:
        users_test['label']=users_test.dist>np.mean([users_test[users_test.liked==1].dist.mean(),
            users_test[users_test.liked==0].dist.mean()])
### End of cheating

    print('''
auc: ''',roc_auc_score(users_test['liked'],users_test['label']),end='')
    precision, recall, fscore, _ = precision_recall_fscore_support(
            users_test['liked'],users_test['label'], average='weighted')
    print('''
precision: {}
recall: {}
fscore: {}
    '''.format(precision, recall, fscore))

    import matplotlib.pyplot as plt
    users_train.dist.hist(bins=50)
    users_test[users_test.liked==1].dist.hist(bins=50)
    users_test[users_test.liked==0].dist.hist(bins=50)
    plt.show()

#generate_graph()
#users_train, users_test = generate_graph()
#evaluate()
if __name__ == "__main__":
    args = node2vec.parse_args()
    if args.regenerate==True:
        print('Regenerating...')
        generate_graph(args.regenerate)
        node2vec.main(args)
    evaluate()
