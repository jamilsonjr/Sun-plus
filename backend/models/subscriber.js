
const EVSchema = require('./EV.js')
const mongoose = require('mongoose')

const subscriberSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true
    },
    subscriberEVs: {
        type: [EVSchema],
        required: false
    },
    subscriberCreditCards: {
        type: Object,
        required: false
    },
    subscribeDate: {
        type: Date,
        required: true,
        default: Date.now
    }
})


module.exports = mongoose.model('Subscriber', subscriberSchema)