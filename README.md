# channeltest

## Usage
`channeltest <frequency multiplier> <symbol width> <symbol size>`

## TODO:
* aggregate frames so that we can seek to the strongest position. This would make single frame syncs a lot easier
* better smoothing to allow lower rates
* allow non-2^n buffer sizes for closer tweaking

## Tweaking
Generally aim for powers of 2, because that's what I've tested.

For low noise environments, try cranking up the frequency multiplier while reducing the symbol width. 
On my system, I get 100% accuracy at about 640 baud from `1 3 256`, but see if you can beat that ;).
Bear in mind that to get single width accuracy, you will need to relaunch a few times to try to sync the buffers.

For higher noise environments, a longer symbol width, and a higher symbol size, will give you more accuracy
at the expense of speed. The frequency multiplier should be chosen based on the frequency response of your mic/speaker,
but fiddle around with it to see what works for you. I get decent results with `4 1 1024`, but YMMV.

If your data is being corrupted, you should see one of two symptoms whilst transmitting:
* Triangles in the symbol window: Your frequencies are too close together, try changing the freq multiplier and the symbol size
* Carrier/data close to sync: You have a lot of background noise, change the frequency, increase the width and size, or go somewhere quieter

If it is neither of these, then please report it, with an audio clip attached of the output.
