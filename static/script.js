//obtendo a url do servidor
const url = 'http://127.0.0.1:5000/test'

//selecionando os elementos do DOM
const chatInput = document.querySelector(".chat__input")
const messageBtn = document.querySelector(".text__chat__button")
const  message = document.querySelector(".text__chat")

// adicionando o evento de click no botão de enviar
messageBtn.addEventListener("click", addMessage)
message.addEventListener("keyup", (e) => {
    if (e.keyCode === 13 ) {
        addMessage()
    }
})

// função para adicionar a mensagem do usuário e a resposta do servidor
async function addMessage() {
    let userMessage = message.value
    
    // verificando se o input está vazio
    if (message.value == "") {
        message.style.border = "1px solid red"
        message.style.boxShadow = "0 0 3px red"
        setTimeout(() => {
            message.style.border = "1px solid #ccc"
            message.style.boxShadow = "none"
        }, 500)
        return
    }

    // limpando o input
    message.value = ""

    // desabilitando o input
    chatInput.style.opacity = ".3"
    chatInput.style.pointerEvents = "none"
    chatInput.style.cursor = "not-allowed"
    document.querySelector(".text__chat").blur()
    

    // removendo a mensagem de demo
    const demoDiv = document.querySelector(".chat__demo")
    if (demoDiv) {
        demoDiv.remove()
    }

    // adicionando a mensagem do usuário
    let chatDiv = document.querySelector(".chat__messages")
    newDiv = 
    `
        <div class="chat__message">
            <div class="icon__user icon">
                <i class='bx bxs-user'></i>
            </div>
            <div class="chat__message__text">
                <span class="text">
                    ${userMessage}
                </span>
            </div>
        </div>
        
    `
    chatDiv.innerHTML += newDiv

    const textUser = document.querySelectorAll(".chat__message__text")
    textUser[textUser.length - 1].scrollIntoView()
    if (textUser.length > 1) {
        textUser[textUser.length - 2].style.animation = "none"
    }
    textUser[textUser.length - 1].style.animation = "popUp .3s ease-in-out"

    // obtendo a resposta do servidor
    message.addEventListener('submit', (e)=> {
        e.preventDefault()
    })
    sendMessage(userMessage).then((res, e) => {
        
        newDiv =
        `
            <div class="chat__message bot__msg">
                <div class="icon__bot icon">
                    <i class='bx bxs-bot'></i>
                </div>
                <div class="chat__message__text__bot">
                    <span class="text"></span>
                    <span class="cursor">.</span>
                </div>
            </div>
        `
        chatDiv.innerHTML += newDiv

        
        // adicionando efeito de typing na resposta do servidor
        let text = res
        // let textArray = text.split("")
        // let texto = ""
        // text.map((item, index) => {
        //     if (index == text.length - 1) {
        //         texto += item
        //         return texto
        //     }
        //     return texto += item + '¬';
        // })
        // text = texto.split('¬')

        
        let textBot = document.querySelectorAll(".text")
        let divBot = document.querySelectorAll(".bot__msg")
        let count = 0
        try {
            text.forEach((letter, i) => {
                let link = ''
                let textArray = letter.text.split("")
                
                textArray.map((item, j) => {
                    count += 1
                    setTimeout(() => {
                        if(item == '¬') {
                            textBot[textBot.length - 1].innerHTML += '<br>'
                            textBot[textBot.length - 1].scrollIntoView()
                        } else {
                            textBot[textBot.length - 1].innerHTML += item
                            textBot[textBot.length - 1].scrollIntoView()
                        }
                        
                        if(textArray.length - 1 == j) {
                            if ('link' in letter) {
                                textBot[textBot.length - 1].innerHTML += `<br> <a href="${link}" target="_blank" class="pdf__link">Link para o pdf</a>.`
                            }
                            if(i == text.length - 1){
                                setTimeout(() => {       
                                    document.querySelector('.cursor').remove()
                                    chatInput.style.opacity = "1"
                                    chatInput.style.pointerEvents = "all"
                                    chatInput.style.cursor = "text"
                                    message.focus()
                                }, 1200)
                            } else {
                                textBot[textBot.length - 1].innerHTML += '<br>'+'<br>'
                            }
                        }
                    }, 30 * count)
                })
                
            })
        } catch (error) {
            textBot[textBot.length - 1].innerHTML += 'ERRO: Não foi possível conectar ao servidor.'
            divBot[divBot.length - 1].classList.add('chat__message__error')
            textBot[textBot.length - 1].scrollIntoView()
            
            setTimeout(() => {
                document.querySelector('.cursor').remove()
                chatInput.style.opacity = "1"
                chatInput.style.pointerEvents = "all"
                chatInput.style.cursor = "text"
                divBot[divBot.length - 1].style.animation = "none"
                message.focus()
            }, 1200)
            throw(error)
        }
        
    })
}

// fazendo a requisição para o servidor com fetch
async function sendMessage(message) {
    try {
        let req = await fetch(url, {
            method: 'POST',
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                'message': message
            }),
        })
        message.value = ""
        let res = await req.json()
        return res
    } catch (error) {
        return `ERRO ${error}. Não foi possível conectar ao servidor.`
    }
}

// limpando log ao dar reload
window.addEventListener("beforeunload", function() {
    const userMessage = "clear";
    const xhr = new XMLHttpRequest();
    const url = "http://127.0.0.1:5000/test";
    const params = JSON.stringify({ 'message': userMessage });
  
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-type", "application/json; charset=utf-8");
    xhr.onreadystatechange = function() {
      if (xhr.readyState === 4 && xhr.status === 200) {
        console.log(xhr.responseText);
      }
    };
    xhr.send(params);
  });