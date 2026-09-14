"use strict";

/*
    ============================================
    SOUL AI 0.1
    ============================================

    Пока это frontend-прототип.

    В будущем:

        sendMessage()
              ↓
        /api/chat
              ↓
        Soul AI Backend
              ↓
        AI Model
              ↓
        ответ Soul
*/


// ============================================
// ELEMENTS
// ============================================

const chat = document.getElementById("chat");

const messageForm =
    document.getElementById("messageForm");

const messageInput =
    document.getElementById("messageInput");

const sendButton =
    document.getElementById("sendButton");

const clearButton =
    document.getElementById("clearChat");

const welcome =
    document.getElementById("welcome");


// ============================================
// MEMORY
// ============================================

const STORAGE_KEY = "soul_ai_messages";

let messages =
    JSON.parse(
        localStorage.getItem(STORAGE_KEY) || "[]"
    );


// ============================================
// INIT
// ============================================

loadMessages();


// ============================================
// SEND MESSAGE
// ============================================

messageForm.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();

        const text =
            messageInput.value.trim();

        if (!text) {
            return;
        }

        addMessage("user", text);

        messageInput.value = "";

        resizeInput();

        await soulReply(text);
    }
);


// ============================================
// ENTER
// ============================================

messageInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();

            messageForm.requestSubmit();
        }
    }
);


// ============================================
// AUTO RESIZE TEXTAREA
// ============================================

messageInput.addEventListener(
    "input",
    resizeInput
);


function resizeInput() {

    messageInput.style.height = "auto";

    messageInput.style.height =
        Math.min(
            messageInput.scrollHeight,
            130
        ) + "px";
}


// ============================================
// SUGGESTIONS
// ============================================

document.addEventListener(
    "click",
    function (event) {

        const button =
            event.target.closest(".suggestion");

        if (!button) {
            return;
        }

        messageInput.value =
            button.textContent.trim();

        resizeInput();

        messageInput.focus();
    }
);


// ============================================
// ADD MESSAGE
// ============================================

function addMessage(role, text, save = true) {

    if (welcome) {
        welcome.style.display = "none";
    }

    const row =
        document.createElement("div");

    row.className =
        `message-row ${role}`;

    const message =
        document.createElement("div");

    message.className = "message";

    message.textContent = text;

    row.appendChild(message);

    chat.appendChild(row);

    scrollToBottom();

    if (save) {

        messages.push({
            role: role,
            text: text,
            time: Date.now()
        });

        saveMessages();
    }
}


// ============================================
// LOAD MESSAGES
// ============================================

function loadMessages() {

    if (messages.length === 0) {
        return;
    }

    welcome.style.display = "none";

    messages.forEach(function (item) {

        addMessage(
            item.role,
            item.text,
            false
        );
    });

    scrollToBottom();
}


// ============================================
// SAVE
// ============================================

function saveMessages() {

    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(messages)
    );
}


// ============================================
// SCROLL
// ============================================

function scrollToBottom() {

    requestAnimationFrame(function () {

        chat.scrollTop =
            chat.scrollHeight;
    });
}


// ============================================
// TYPING INDICATOR
// ============================================

function showTyping() {

    const row =
        document.createElement("div");

    row.className =
        "message-row soul";

    row.id = "typingIndicator";

    const typing =
        document.createElement("div");

    typing.className =
        "message typing";

    typing.innerHTML = `
        <span></span>
        <span></span>
        <span></span>
    `;

    row.appendChild(typing);

    chat.appendChild(row);

    scrollToBottom();
}


function hideTyping() {

    const typing =
        document.getElementById(
            "typingIndicator"
        );

    if (typing) {
        typing.remove();
    }
}


// ============================================
// SOUL RESPONSE
// ============================================

async function soulReply(userText) {

    sendButton.disabled = true;

    showTyping();

    /*
        Имитация задержки ответа.

        Позже здесь будет:

        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: userText
            })
        });

        const data = await response.json();

        addMessage("soul", data.message);
    */

    await sleep(
        700 + Math.random() * 700
    );

    hideTyping();

    const answer =
        generateDemoResponse(userText);

    addMessage(
        "soul",
        answer
    );

    sendButton.disabled = false;

    messageInput.focus();
}


// ============================================
// TEMPORARY AI
// ============================================

function generateDemoResponse(text) {

    const lower =
        text.toLowerCase();

    if (
        lower.includes("привет") ||
        lower.includes("здравств")
    ) {
        return "Привет 👋 Я Soul AI. Пока я нахожусь на стадии 0.1, но мы постепенно создадим настоящий интеллект.";
    }

    if (
        lower.includes("кто ты") ||
        lower.includes("расскажи о себе")
    ) {
        return "Я — Soul AI. Это наш первый прототип. Сейчас у меня ещё нет настоящей языковой модели, но следующим этапом мы подключим мой мозг.";
    }

    if (
        lower.includes("что ты умеешь")
    ) {
        return "Пока я умею общаться через этот интерфейс и хранить историю чата. Дальше добавим память, настоящий AI, голос, зрение и инструменты.";
    }

    if (
        lower.includes("начн") ||
        lower.includes("поехали")
    ) {
        return "Поехали 🚀 Soul AI начинается именно здесь.";
    }

    return `Я получил твоё сообщение: «${text}»\n\nСейчас я ещё прототип. Следующим этапом мы подключим настоящий интеллект Soul AI.`;
}


// ============================================
// CLEAR CHAT
// ============================================

clearButton.addEventListener(
    "click",
    function () {

        const confirmed =
            confirm(
                "Очистить историю Soul AI?"
            );

        if (!confirmed) {
            return;
        }

        messages = [];

        localStorage.removeItem(
            STORAGE_KEY
        );

        document
            .querySelectorAll(".message-row")
            .forEach(function (element) {
                element.remove();
            });

        welcome.style.display = "flex";

        messageInput.focus();
    }
);


// ============================================
// UTILITY
// ============================================

function sleep(milliseconds) {

    return new Promise(
        function (resolve) {

            setTimeout(
                resolve,
                milliseconds
            );
        }
    );
}