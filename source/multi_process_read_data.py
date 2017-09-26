import bson                       # this is installed with the pymongo package
import multiprocessing as mp      # will come in handy due to the size of the data
import time
from multiprocessing import cpu_count



 # note the difference
def process(q, iolock,all_ids,all_categories,all_imgs,all_weights):
    i=0
    while True:
        d = q.get()
        if d is None:
            break
        product_id = d['_id']
        category_id = d['category_id']
        weight = 1/len(d['imgs'])
        for e, pic in enumerate(d['imgs']):
            all_ids.append(product_id)
            all_categories.append(category_id)
            all_imgs.append(pic['picture'])
            all_weights.append(weight)



def load_train_data(path,cutoff):
    NCORE =  cpu_count()
    prod_to_category = mp.Manager().dict()
    all_categories = mp.Manager().list()
    all_ids = mp.Manager().list()
    all_imgs = mp.Manager().list()
    all_weights = mp.Manager().list()

    q = mp.Queue(maxsize=NCORE)
    iolock = mp.Lock()
    pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock, all_ids, all_categories, all_imgs, all_weights))


    # process the file

    data = bson.decode_file_iter(open(path, 'rb'))
    it=0
    for c, d in enumerate(data):
        if it>=cutoff:
            break
        q.put(d)  # blocks until q below its max size
        it=it+1

    # tell workers we're done

    for _ in range(NCORE):
        q.put(None)
    pool.close()
    pool.join()

    # convert back to normal dictionary
    all_categories = list(all_categories)
    all_ids = list(all_ids)
    all_imgs = list(all_imgs)
    all_weights = list(all_weights)
    return all_categories,all_ids,all_imgs,all_weights


def get_batches(ids, imgs, categories, weights, batch_size):
    n_batches = len(ids)//batch_size
    ids, imgs, categories, weights = ids[:n_batches*batch_size], imgs[:n_batches*batch_size], categories[:n_batches*batch_size], weights[:n_batches*batch_size]
    for ii in range(0, len(ids), batch_size):
        yield ids[ii:ii+batch_size], imgs[ii:ii+batch_size], categories[ii:ii+batch_size], weights[ii:ii+batch_size]

def get_splitted_data(ids, imgs, categories, weights, test_size, val_size):
    from sklearn.model_selection import train_test_split
    remain_ids, test_ids, remain_imgs, test_imgs, remain_categories, test_categories, remain_weights, test_weights =  train_test_split(
        ids, imgs, categories, weights, test_size = test_size)
    train_ids, val_ids, train_imgs, val_imgs, train_categories, val_categories, train_weights, val_weights = train_test_split(
        remain_ids, remain_imgs, remain_categories, remain_weights, test_size=val_size/(1-test_size))
    new_ids = {'train':train_ids, 'val':val_ids, 'test':test_ids}
    new_imgs = {'train':train_imgs, 'val':val_imgs, 'test':test_imgs}
    new_categories = {'train':train_categories, 'val':val_categories, 'test':test_categories}
    new_weights = {'train':train_weights, 'val':val_weights, 'test':test_weights}
    return new_ids, new_imgs, new_categories, new_weights

def auto_load_three_sets(path, cutoff):
    t1 = time.time()
    print('loading data')
    all_categories, all_ids, all_imgs, all_weights = load_train_data(path,cutoff)
    t2 = time.time()
    n_unique_cat = len(set(all_categories))
    print(str(cutoff)+' lines of data loaded. Took '+str(round(t2-t1,3))+'s to load.')
    print('Total number of images: '+str(len(all_ids)))
    print('Total number of unique category_ids: '+str(n_unique_cat))

    print('Splitting train/val/test set')
    ids, imgs, categories, weights = get_splitted_data(all_ids, all_imgs, all_categories, all_weights, test_size=0.2, val_size=0.1)
    del all_ids, all_imgs, all_categories, all_weights

    t3=time.time()
    print('Train/Val/Test splitted. Took time: '+str(round(t3-t2,3))+'s')
    print('Train set size: '+str(len(ids['train'])))
    print('Val set size: '+str(len(ids['val'])))
    print('Test set size: '+str(len(ids['test'])))
    return ids, imgs, categories, weights
if __name__ == '__main__':
    cutoff = 1000
    ids, imgs, categories, weights = auto_load_three_sets('../input/',cutoff)
