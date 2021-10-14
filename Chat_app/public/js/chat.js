

const socket = io()

const $messageForm = document.querySelector('#message-form')
const $messageFormInput = $messageForm.querySelector('input')
const $messageFormButton = $messageForm.querySelector('button')
const $sendLocationButton = document.querySelector('#send-location')
const $messages = document.querySelector('#messages')

const messageTemplate = document.querySelector('#message-template').innerHTML
const sidebarTemplate = document.querySelector('#sidebar-template').innerHTML
const { username , room} = Qs.parse(location.search, { ignoreQueryPrefix: true })


const autoscroll = () => {
   
    //new message element
    const $newMessage = $messages.lastElementChild
    //height of new message
    const newMessageStyles = getComputedStyle($newMessage)
    const newMessageMargin = parseInt(newMessageStyles.marginBottom)
    const newMessageHeight = $newMessage.offsetHeight + newMessageMargin

    console.log(newMessageHeight)

    //visible height
    const visibleHeight = $messages.offsetHeight
    //height of container
    const containerHeight = $messages.scrollHeight
    console.log(containerHeight)
    const scrollOffset = $messages.scrollTop + visibleHeight
    const xx=containerHeight - newMessageHeight;
    console.log(xx)
    console.log(scrollOffset)
    if (Math.round(containerHeight - newMessageHeight) >= Math.round(scrollOffset)) {
        $messages.scrollTop=$messages.scrollHeight
        console.log("scroll bolltom")
    }
}
socket.on('message', (message) => {
    console.log(message)
    const html = Mustache.render(messageTemplate, {
        username: message.username,
        message: message.text,
        createdAt: moment(message.createdAt).format('h:mm a')
    })
    $messages.insertAdjacentHTML('beforeend', html)
    autoscroll()

})



socket.on('roomData', ({ room, users }) => {
    const html = Mustache.render(sidebarTemplate, {
        room,
        users
    })
    document.querySelector('#sidebar').innerHTML=html
})
$messageForm.addEventListener('submit', (e) => {
    e.preventDefault()
    $messageFormButton.setAttribute('disabled','disabled')
    const message = e.target.elements.message.value
    socket.emit('sendMessage', message, (error) => {
        $messageFormButton.removeAttribute('disabled')
        $messageFormInput.value = ''
        $messageFormInput.focus()

        if (error) {
            console.log(error)
        }

        console.log('the message is deliverd', message)
    })
})




/*
socket.on('countUpdated', (count) => {
console.log('the count has been updated'+count)
})

document.querySelector('#increment').addEventListener('click', () => {
    console.log('clicked')
    socket.emit('increment')
})*/

socket.emit('join', { username, room }, (error) => {
    if (error) {
        alert(error)
        location.href = '/'
    }
})