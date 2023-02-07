# Import the necessary modules
import tensorflow as tf
from tensorflow import keras

# 1. Load the pre-trained model
model = keras.models.load_model("path/to/model.h5")

# 2. Get the last layer of the model (remove softmax)
last_layer = model.layers[-1]

# Remove the last layer from the model
model.pop()

# (*) ECE result before calibration
import tensorflow_probability as tfp

num_bins = 50
labels_true = tf.convert_to_tensor(y_val, dtype=tf.int32, name='labels_true')
logits = tf.convert_to_tensor(y_pred, dtype=tf.float32, name='logits')

tfp.stats.expected_calibration_error(num_bins=num_bins, 
                                     logits=logits, 
                                     labels_true=labels_true)
# 3. Temperature Scaling

temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32) 

def compute_loss():
    y_pred_model_w_temp = tf.math.divide(y_pred, temp)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                tf.convert_to_tensor(y_val), y_pred_model_w_temp))
    return loss

optimizer = tf.optimizers.Adam(learning_rate=0.01)

print('Temperature Initial value: {}'.format(temp.numpy()))

for i in range(300):
    opts = optimizer.minimize(compute_loss, var_list=[temp])


print('Temperature Final value: {}'.format(temp.numpy()))


# (*) ECE result after calibration
y_pred_model_w_temp = tf.math.divide(y_pred, temp)
num_bins = 50
labels_true = tf.convert_to_tensor(y_val, dtype=tf.int32, name='labels_true')
logits = tf.convert_to_tensor(y_pred_model_w_temp, dtype=tf.float32, name='logits')

# 4. add layer
final_model = keras.models.Model(inputs=model.inputs,
                                  outputs=TempScaling(temperature=best_temperature)(model.outputs))

final_model.add(keras.layers.Dense(last_layer.output_shape[1], activation='softmax'))

# 5. Compile the model
final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


##########################################################
calibration_graph(preds[:,1], probs[:,1], y_val[:,1]) 

def calibration_graph(preds, scores, trues):
    preds= np.array(preds)  
    scores = np.array(scores)  
    true = np.array(trues)  
    num_data = len(scores)
    interval = 0.1
    num_intervals = int(1/interval)
    
    IntervalRange = collections.namedtuple('IntervalRange', 'start end')
    intervals=[]
    for i in range(num_intervals):
        _intev = IntervalRange(start=i*interval, end=(i+1)*interval)
        intervals.append(_intev)
        
    percents = []
    accs = []
    eces = []
    for itv in intervals:
        interval_scores = scores[np.where((scores >= itv.start)&(scores < itv.end), True, False)]
        interval_preds = preds[np.where((scores >= itv.start)&(scores  < itv.end), True, False)]
        interval_trues = true[np.where((scores >= itv.start)&(scores < itv.end), True, False)]

        percent = len(interval_scores) / num_data
        if list(interval_scores):
            acc = np.sum(np.equal(interval_trues, interval_preds)) / len(interval_trues)
            conf = np.average(scores)

            percents.append(percent)
            accs.append(acc)
            eces.append((len(interval_scores)/num_data)*(np.abs(acc-conf)))
        else:
            percents.append(0.0)
            accs.append('nan')
            
            
    avg_acc = np.sum(np.equal(preds, trues))/len(trues)

    plt.figure(figsize=(12,6))
    plt.xlabel('confidence')
    
    plt.subplot(1,2,1)
    plt.bar([intv.start for intv in intervals], percents,width=0.1, color='lightskyblue', edgecolor='silver', align='edge', alpha=0.6)
    plt.grid(True,alpha=0.7, linestyle='--')
    plt.ylim(0,1)
    plt.axvline(x=avg_acc, linestyle='--', color='b')
    plt.text(avg_acc-0.08,0.5,'accuracy',rotation=90, color='b')
    plt.axvline(x=np.average(scores), linestyle='--', color='g')
    plt.text(np.average(scores)+0.03,0.4,'avg. confidence',rotation=90, color='g')
    plt.ylabel('% of samples', fontsize='x-large')
    plt.xlabel('confidence', fontsize='x-large')
    plt.title('confidence histogram', fontsize='xx-large')

    plt.subplot(1,2,2)
    text = 'Error=%.4f'%np.sum(eces)
    plt.bar([intv.start for intv in intervals], [intv.end for intv in intervals], width=0.1, color='plum', edgecolor='silver',  align='edge', alpha=0.5, label='Ideal')
    plt.bar([intv.start for intv in intervals], [a if a != 'nan' else 0.0 for a in accs ],width=0.1, color='lightskyblue', edgecolor='silver', align='edge', alpha=0.6, label='Outputs')
    plt.grid(True,alpha=0.7, linestyle='--')
    plt.legend()
    plt.text(0.6,0.05,text, backgroundcolor='w', alpha=0.6)
    plt.ylabel('accuracy', fontsize='x-large')
    plt.xlabel('confidence', fontsize='x-large')
    plt.title('reliability diagram', fontsize='xx-large')
    
    plt.subplots_adjust(wspace=0.25)
    
    plt.show()

###########################################################
input_layer = keras.Input(shape=(5000, 8)) # 5000 vectors of 8-dimensinal vectors
x = keras.layers.Conv1D(filters=32, padding = 'same', kernel_size=7, dilation_rate=1)(input_layer)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.Conv1D(filters=32, padding = 'same', kernel_size=5)(x)
x = keras.layers.Conv1D(filters=16, padding = 'same', kernel_size=5, dilation_rate=2)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPool1D(pool_size = 2)(x)
x = keras.layers.Conv1D(filters=32, padding = 'same', kernel_size=5)(x)

x = keras.layers.Conv1D(filters=16, padding = 'same', kernel_size=5, dilation_rate=4)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPool1D(pool_size = 4)(x)
x = keras.layers.Conv1D(filters=32, padding = 'same', kernel_size=5)(x)

x = keras.layers.Conv1D(filters=64, padding = 'same', kernel_size=5, dilation_rate=8)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPool1D(pool_size = 2)(x)
x = keras.layers.Conv1D(filters=64, padding = 'same', kernel_size=5)(x)

x = keras.layers.Conv1D(filters=64, padding = 'same', kernel_size=3, dilation_rate=16)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)

x = keras.layers.Conv1D(filters=128, padding = 'same', kernel_size=12, dilation_rate=32)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Activation('relu')(x)
x = keras.layers.MaxPool1D(pool_size = 2)(x)

x = keras.layers.GlobalAveragePooling1D()(x)

x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
logits = keras.layers.Dense(2)(x)
model = keras.Model(inputs=input_layer, outputs=logits)


custom_loss = keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(loss= custom_loss, 
                 optimizer='adam',
                 metrics=['accuracy'])


sval_yi = 0.014299999922513962
sval_pred = (sval_proba > sval_yi).astype(np.int64)