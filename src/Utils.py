"""
class UNet(tf.keras.Model):
    def __init__(self,out = 16, final_layer=21, kernel=(3,3)):
        super(UNet, self).__init__()

        self.out = out
        self.final_layer = final_layer

        self.ConvBlock1 = ConvBlock(out=out, kernel = kernel)
        self.ConvBlock2 = ConvBlock(out=out*2, kernel = kernel)
        self.ConvBlock3 = ConvBlock(out=out*4, kernel = kernel)
        self.ConvBlock4 = ConvBlock(out=out*8, kernel = kernel)
        self.ConvBlock5 = ConvBlock(out=out*4, kernel = kernel)
        self.ConvBlock6 = ConvBlock(out=out*2, kernel = kernel)
        self.ConvBlock7 = ConvBlock(out=out, kernel = kernel)
        self.ConvBlock8 = ConvBlock(out=final_layer, kernel = kernel)

        self.maxpool = keras.layers.MaxPool2D((2,2))
        self.upsample = keras.layers.UpSampling2D((2,2))

    def call(self, inputs):
        inputs = keras.Input(shape=inputs)

        #block 1
        X1 = self.ConvBlock1(inputs)
        X_pool = self.maxpool(X1)

        # block 2
        X2 = self.ConvBlock2(X_pool)
        X_pool = self.maxpool(X2)

        # block 3
        X3 = self.ConvBlock3(X_pool)
        X_pool = self.maxpool(X3)

        # block 4
        X4 = self.ConvBlock4(X_pool)
        X_pool = self.upsample(X4)


        # block 5
        X = tf.concat([X_pool, X3], axis = -1)
        X = self.ConvBlock5(X)
        X = self.upsample(X)

        # block 6
        X = tf.concat([X, X2], axis = -1)
        X = self.ConvBlock6(X)
        X = self.upsample(X)

        # block 7
        X = tf.concat([X, X1], axis = -1)
        X = self.ConvBlock7(X)

        # output
        X = self.ConvBlock8(X)
        out = tf.keras.activations.sigmoid(X)

        model = keras.Model(inputs, out)

        return model


class ConvBlock(tf.keras.Model):
  def __init__(self, out=3, kernel=(3,3)):
      super(ConvBlock, self).__init__()
      self.conv1 = keras.layers.Conv2D(filters=out, kernel_size = kernel, activation=activations.relu, padding='same')
      self.batchnorm = keras.layers.BatchNormalization()
      self.conv2 = keras.layers.Conv2D(filters=out, kernel_size=kernel, activation=activations.relu, padding='same')
      self.batchnorm2 = keras.layers.BatchNormalization()

  def call(self, inputs):
      X = self.batchnorm(inputs)
      X = self.conv1(X)
      X = self.batchnorm2(X)
      out = self.conv2(X)
      return out"""


