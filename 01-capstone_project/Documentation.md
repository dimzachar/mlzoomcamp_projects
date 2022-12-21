## Documentation

* build_model function

I used the following architecture for a CNN model used for image classification tasks. 

It begins with a pre-trained model Xception from Keras Applications, which serves as the base model for the architecture. The base model's weights are initialized with the 'imagenet' weights and its top layers are not included in the model. The input shape of the model is specified as a tuple of values, which defines the size and number of channels of the input data (e.g. (150, 150, 3) for images with given resolution and 3 color channels).

The base model is set to be not trainable, meaning that its weights will not be updated during the training process. The model then takes in an input with the specified shape and passes it through the base model, setting the 'training' parameter to False to indicate that the base model should not be updated.

The output of the base model is then passed through a global average pooling layer, which reduces the spatial dimensions of the output while preserving the channel dimensions. This is followed by a dense layer with a specified number of units and a ReLU activation function.

The model also includes an optional dropout layer with a specified dropout rate, which randomly sets a fraction of the inputs to zero during training in order to prevent overfitting. The output of the dropout layer (or the output of the dense layer if the dropout layer is not included) is then passed through a final dense layer with a single unit and a sigmoid activation function, which outputs a probability value between 0 and 1.

The model is then compiled with an Adam optimizer and a binary cross-entropy loss function, and its performance is evaluated using accuracy metrics. The model's summary is also printed to provide a summary of the model's layers and their sizes.


<h1>checkpoint_weight function</h1>


<p>The <code>checkpoint_weights</code> function is a utility function that sets up several callbacks for use during model training. The main purpose of the callbacks is to save the best model to a file during training and to stop the training if the validation accuracy does not improve after a certain number of epochs. These callbacks include:</p>
<ul>
  <li><code>ModelCheckpoint</code>: This callback saves the best model weights to a file, with a name that includes the epoch number and the validation accuracy.</li>
  <li><code>EarlyStopping</code>: This callback stops the training if the validation accuracy does not improve after a specified number of epochs (in this case, 3 epochs).</li>
  <li><code>TensorBoard</code>: This callback is used to visualize the training progress in TensorBoard, a tool for analyzing and debugging machine learning models.</li>
</ul>
<h3>Usage</h3>
<p>To use the <code>checkpoint_weights</code> function, you need to provide the following arguments:</p>
<ul>
  <li><code>model_name</code>: The name of the model. This name will be used to name the checkpoint files that are saved.</li>
  <li><code>checkpoint_dir</code>: The directory where the model checkpoint files will be saved.</li>
  <li><code>log_dir</code>: The directory where the TensorBoard logs will be saved.</li>
  <li><code>delete_files</code> (optional, default=True): A flag to indicate whether to delete existing checkpoints. If this flag is set to <code>True</code>, the function will delete all files in the <code>checkpoint_dir</code> directory that contain the <code>model_name</code> in their names. If this flag is not set or is set to <code>False</code>, the function will append an underscore and a number to the end of <code>model_name</code> if there are any files in the <code>checkpoint_dir</code> directory that contain <code>model_name</code> in their names.</li>
  <li><code>restore_from_checkpoint</code> (optional, default=False): A flag to indicate whether to restore the model weights from a checkpoint file. If this flag is set to <code>True</code>, the function will find the latest checkpoint file in the <code>checkpoint_dir</code> directory and load the model weights from it.</li>
  <li><code>callbacks</code> : a list of callbacks to use during training. If this parameter is not provided, the function will create the ModelCheckpoint, EarlyStopping, and TensorBoard callbacks.</li>

</ul>
<p>The function first checks if the <code>checkpoint_dir</code> and <code>log_dir</code> directories exist, and creates them if they do not. It then checks the value of the <code>restore_from_checkpoint</code> flag and, if it is set to True, searches for the latest checkpoint file in the <code>checkpoint_dir</code> directory with the same model name and restores the model weights from it.</p>
<p>The function then checks the value of the <code>delete_files</code> flag. If it is set to True, the function deletes all existing checkpoint files with the same model name in the <code>checkpoint_dir</code> directory. If <code>delete_files is not set or is set to False, the function appends an underscore and a number to the end of <code>model_name</code> if there are any existing checkpoint files with the same model name in the <code>checkpoint_dir</code> directory. The number is equal to the number of such files in the directory.</p>
<p>The function then creates a <code>ModelCheckpoint</code> callback and specifies the <code>checkpoint_dir</code> directory and the <code>model_name as the file name for the checkpoint file. The callback is set to save the best model. It is also set to monitor the validation accuracy and save the model when the validation accuracy improves.</p>
<p>The function also creates an <code>EarlyStopping</code> callback and specifies the number of epochs to wait before stopping the training if the validation accuracy does not improve. Finally, the function creates a <code>TensorBoard</code> callback and specifies the <code>log_dir</code> directory as the location for the <code>TensorBoard</code> logs.</p>

<p>The <code>checkpoint_weights</code> function returns a list containing the <code>ModelCheckpoint</code>, <code>EarlyStopping</code>, and <code>TensorBoard</code> callbacks, which you can pass to the <code>fit</code> function of your model to use during training.</p>
<p>Here is an example of how to use the <code>checkpoint_weights</code> function to set up the callbacks for model training:</p>
<pre><code>callbacks = checkpoint_weights(model_name='my_model', checkpoint_dir='checkpoints', log_dir='logs')

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=callbacks)
</code></pre>
<p>This will train the model using the provided training and validation data, and save the best model weights to a file in the <code>checkpoints</code> directory with a name that includes the epoch number and the validation accuracy. If the validation accuracy does not improve after 3 epochs, the training will be stopped and the best model weights will be restored. In addition, the training progress will be logged to the <code>logs</code> directory and can be visualized in TensorBoard.</p>


<h1>train function</h1>

<p>The <code>train</code> function allows you to train and evaluate a model for multiple learning rates.</p> The <code>build_model</code> function is used to construct the model using the supplied learning rate, dropout rate, and other parameters. The model is then trained using the supplied training data and epochs, as well as the specified callbacks. The training history is saved and returned as a dictionary, with the hyperparameters (learning rate and dropout rate) as keys and the training history objects as values. In addition, the function returns the trained model with the highest validation accuracy.

It appears that the model is trained using a static graph defined by the train_step function. The train_step function performs a single training step, using a gradient tape to compute the gradients of the loss function with respect to the model's trainable variables and applying the gradients using the optimizer. The loss value resulting from the training step is returned by the function.

<h2>Parameters:</h2>
<ul>
  <li><code>rates</code>: a list of float values representing the learning rates to use for training the model.</li>
  <li><code>checkpoint_weights</code>: a function that creates a ModelCheckpoint callback for saving the model weights during training.</li>
  <li><code>epochs</code>: an integer representing the number of epochs to train the model.</li>
  <li><code>input_shape</code>: a tuple representing the shape of the input data.</li>
  <li><code>include_dropout</code>: a boolean indicating whether or not to include dropout layers in the model.</li>
</ul>
<h2>Returns:</h2>
<ul>
  <li><code>scores</code>: a dictionary containing the training history for each learning rate. The keys of the dictionary are the learning rates and the values are the training history objects returned by the <code>fit</code> method.</li>
  <li><code>model</code>: a trained <code>tf.keras.Model</code> with the best validation accuracy.</li>
</ul>
<h2>Example</h2>
<pre><code>scores, model = train(rates=[0.001, 0.01, 0.1],
                      callbacks=create_checkpoint_callback,
                      epochs=10,
                      input_shape=(128, 128, 3),
                      include_dropout=True)
</code></pre>

* plot function 

We define a function plot that takes four arguments:

    history: an object containing the training and validation data for a model, typically returned by the fit method of a Keras model.
    label: a string used to label the data in the plots.
    max_epochs: an integer representing the maximum number of epochs to plot.
    fig_num: an integer representing the number of the figure being plotted.

The plot function first extracts the accuracy and loss data from the history object, and then creates two subplots: one for accuracy and one for loss. For each subplot, it plots the training and validation data and saves the figure to a file. The plot function does not show the figure; it only saves it to a file.

After defining the plot function, the code iterates over the items in the scores dictionary and calls the plot function for each item, passing in the corresponding history, label, and figure number.

<h1>plot_image_extension_frequency</h1>

<p>This function plots the frequency of different image extensions in a given directory.</p>

<h2>Usage</h2>

<pre><code>plot_image_extension_frequency(path)
</code></pre>

<h2>Parameters</h2>

<ul>
  <li><code>path</code>: (required) The path to the directory to be scanned.</li>
</ul>

<h2>Example</h2>

<pre><code>plot_image_extension_frequency('./Images/train')
</code></pre>

<h2>Output</h2>

<p>A bar chart showing the frequency of each image extension in the given directory.</p>

<h2>Notes</h2>

<ul>
  <li>The function uses the <code>os.walk</code> function to iterate over the files and subfolders in the given <code>path</code>.</li>
  <li>The function extracts the extension of each file using the <code>os.path.splitext</code> function and adds it to a list called <code>extensions</code>.</li>
  <li>The function uses the <code>collections.Counter</code> function to count the frequency of each extension in the <code>extensions</code> list.</li>
  <li>The function sets a list of colors for the different extensions and uses a <code>for</code> loop to iterate over the unique extensions.</li>
  <li>For each extension, the function plots a bar chart using <code>matplotlib</code>'s <code>bar</code> function. The function uses the <code>extension</code> as the x-axis label and the frequency of the extension as the y-axis label. The color of the bar is chosen from the <code>colors</code> list using the index <code>i</code> modulo the length of the <code>colors</code> list.</li>
  <li>The function adds labels to the x-axis, y-axis, and the chart title using <code>matplotlib</code>'s <code>xlabel</code>, <code>ylabel</code>, and <code>title</code> functions.</li>
  <li>The function displays the plot using <code>matplotlib</code>'s <code>show</code> function.</li>
</ul>