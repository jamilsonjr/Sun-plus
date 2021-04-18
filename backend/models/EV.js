const mongoose = require('mongoose')
const EVSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    brand: {
        type: String,
        required: false
    },
    BatteryMaximumCapacity: {
        type: Number,
        required: true
    },
    StateOfCharge: {
        type: Number,
        required: true
    }
})

module.exports =  EVSchema