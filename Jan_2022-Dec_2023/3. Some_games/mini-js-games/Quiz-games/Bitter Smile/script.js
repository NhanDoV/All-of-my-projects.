const quizData = {
    0: { // Động vật
        name: "🦁 Bộ Đề Động Vật",
        questions: [
            {
                question: "xi jinping / 习近平 giống con gì?",
                image: "images/de1_01.png",
                options: ["Chó", "Lồn", "Gấu", "Cặc"],
                correct: 1
            },
            {
                question: "Con vật nào được liên tưởng đến 江泽民 / Jiāng Zé Mín",
                image: "images/de1_02.png",
                options: ["Đĩ", "Chó", "Bò", "Cặc"],
                correct: 3
            },
            {
                question: "Mao Zedong / 毛泽东 tuổi gì",
                image: "images/de1_03.png",
                options: ["Nhục", "Đĩ", "Lồn", "Chó"],
                correct: 1
            },
            {
                question: "CSGT được giang hồ ví von như",
                image: "images/de1_04.png",
                options: ["Chó đẻ", "Chó vàng", "Cái Lồn", "Con cặc"],
                correct: 1
            },
            {
                question: "Trong hình này có những con gì",
                image: "images/de1_05.png",
                options: ["Bò vàng và bò đỏ", "Bò vàng và chó đỏ", "con chó ăn bò dát vàng", "súc sinh ăn thịt bò"],
                correct: 2
            }            
        ]
    },
    1: { // Địa lý
        name: "🌍 Bộ Đề Địa Lý",
        questions: [
            {
                question: "China bị chặt đầu nhiều nhất bởi người bạn nào",
                image: "images/de2_01.png",
                options: ["Nhật Pỏn", "Mãn Thanh", "Mông Cổ", "Khiết Đan"],
                correct: 1
            },
            {
                question: "China từng là thuộc địa của?",
                image: "images/de2_02.png",
                options: ["Japan", "Mongol", "Western", "Taiwan"],
                correct: 1      // index from options
            },
            {
                question: "Tây Tạng là của",
                image: "images/de2_03.png",                
                options: ["Ấn Độ", "Nepal", "Bangladesh", "Pakistan"],
                correct: 0
            },
            {
                question: "Lá cờ này là của",
                image: "images/de2_04.png",
                options: ["China", "Chinese", "中國 ", "Trung Quốc"],
                correct: 1
            },
            {
                question: "Làng nào trong hoạt hình có tổng bí thư, chủ tịch,.. đều là con vật ngu ngốc",
                image: "images/de2_05.png",
                options: ["Làng Đổn", "Làng `+ Sổn`", "Làng `Công Ôn`", "Cả 3 ngôi làng trên"],
                correct: 3 
            },
        ]
    },
    2: { // Ẩm thực
        name: "🍎 Bộ Đề Ẩm Thực",
        questions: [
            {
                question: "Món ăn nào là bốc mùi nhất của tàu khựa?",
                image: "images/de3_01.png",
                options: ["Đậu hủ thúi", "Cứt tươi Luosifen", "Bánh bao thịt người", "Nước đái đồng tử"],
                correct: 1
            },
            {
                question: "Trung Quốc thích món ăn nào nhất",
                image: "images/de3_02.png",
                options: ["Cứt tươi sông Dương Tử", "Nước đái bò sông Hằng", "Nước thải công nghiệp", "Tất cả đều đúng"],
                correct: 3
            },
            {
                question: "Lũ óc chó du khách Trung Quốc hay làm gì khi ra nước ngoài",
                image: "images/de3_03.png",
                options: ["Làm đĩ", "Làm chó đái bậy", "Ăn cướp", "Tất cả đều đúng"],
                correct: 3
            }
        ]
    },
    3: { // Khoa học
        name: "🔬 Bộ Đề Khoa Học",
        "questions": [
            {
                "question": "Năm 2013, Mỹ cáo buộc Trung Quốc đánh cắp công nghệ quốc phòng và báo cáo về tổ chức nào",
                image: "images/de4_01.png",
                "options": ["Sao Hỏa", "bảo tàng 798", "bệnh viện 731", "đơn vị 61398 PLA"],
                "correct": 3
            },
            {
                "question": "Năm 2018 Mỹ truy tố bao nhiêu điệp viên Trung Quốc ăn cắp công nghệ hàng không GE Aviation?",
                image: "images/de4_02.png",
                "options": ["5", "10", "15", "20"],
                "correct": 1
            },
            {
                "question": "2018, báo cáo FBI và Văn phòng Đại diện Thương mại Mỹ công khai Trung Quốc đánh cắp tài sản trí tuệ, bao gồm mã nguồn phần mềm và công nghệ vũ khí; gây thiệt hại bao nhiêu?",
                image: "images/de4_03.png",
                "options": ["225-600 tỷ USD/năm", "25-60 tỷ USD/năm", "2250-6000 tỷ USD/năm", "2-6 tỷ USD/năm"],
                "correct": 0
            },
            {
                "question": "2021, cty nào bị nghi vấn liên quan hack Nortel (Canada), đánh cắp tài liệu nội bộ dẫn đến sụp đổ công ty; cùng cáo buộc sao chép công nghệ quân sự từ Mỹ/Nga",
                image: "images/de4_04.png",
                "options": ["Huawei", "Zalo", "Alibaba", "VinGroup"],
                "correct": 0
            },
        ]
    },
    4: { // Âm nhạc
        name: "🎵 Bộ Đề Âm Nhạc",
        questions: [
            {
                question: "Đàn nào của Trung Quốc được chơi nhiều nhất trên thế giới",
                image: "images/de5_01.png",
                options: ["đàn bà", "đàn pà", "đàn bầu", "đàn lồng"],
                correct: 1
            },
            {
                question: "Thể loại nhạc nào của Trung Quốc hay được remix nhiều nhất trên TikTok?",
                image: "images/de5_02.png",
                options: ["Nhạc cổ phong", "Nhạc phụ", "Nhạc mẫu", "Cả 3 đáp án"],
                correct: 0
            },
            {
                question: "Đặc điểm dễ nhận ra nhất của nhạc phim Trung Quốc là gì?",
                image: "images/de5_03.png",
                options: [
                    "Bài dài 7-10 phút",
                    "Điệp khúc lặp lại liên tục",
                    "Nghe là thấy cảnh quay chậm",
                    "Tất cả đáp án trên"
                ],
                correct: 3
            }
        ]
    }
};

let currentSet = -1;
let currentQuestion = 0;
let score = 0;
let answered = false;

const screens = {
    menu: document.getElementById('menu-screen'),
    game: document.getElementById('game-screen'),
    end: document.getElementById('end-screen')
};

document.querySelectorAll('.set-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        currentSet = parseInt(btn.dataset.set);
        startQuiz();
    });
});

function startQuiz() {
    currentQuestion = 0;
    score = 0;
    screens.menu.classList.remove('active');
    screens.game.classList.add('active');
    showQuestion();
}

function showQuestion() {
    const quiz = quizData[currentSet];
    const q = quiz.questions[currentQuestion];
    
    document.getElementById('set-name').textContent = quiz.name;
    const total = quiz.questions.length;
    document.getElementById('question-progress').textContent =
        `Câu ${currentQuestion + 1}/${total}`;

    document.getElementById('score').textContent = score;
    
    const imgDiv = document.getElementById('question-image');
    if (q.image) {
        imgDiv.innerHTML = `<img src="${q.image}" alt="Hình câu hỏi">`;
    } else {
        imgDiv.innerHTML = '❓';
    }
    
    document.getElementById('question-text').textContent = q.question;
    
    const optionsDiv = document.getElementById('options');
    optionsDiv.innerHTML = '';
    q.options.forEach((option, index) => {
        const btn = document.createElement('div');
        btn.className = 'option';
        btn.textContent = option;
        btn.dataset.index = index;
        btn.addEventListener('click', () => selectOption(index, q.correct));
        optionsDiv.appendChild(btn);
    });
    
    document.getElementById('next-btn').style.display = 'none';
    answered = false;
}

function selectOption(selected, correct) {
    if (answered) return;
    
    answered = true;
    const options = document.querySelectorAll('.option');
    
    options.forEach((opt, index) => {
        if (index === correct) {
            opt.classList.add('correct');
        } else if (index === selected && selected !== correct) {
            opt.classList.add('incorrect');
        }
        
        opt.style.pointerEvents = 'none';
    });
    
    if (selected === correct) {
        score++;
        document.getElementById('score').textContent = score;
    }
    
    setTimeout(() => {
        document.getElementById('next-btn').style.display = 'block';
    }, 1000);
}

document.getElementById('next-btn').addEventListener('click', nextQuestion);

function nextQuestion() {
    currentQuestion++;
    const total = quizData[currentSet].questions.length;

    if (currentQuestion < total) {
        showQuestion();
    } else {
        endQuiz();
    }
}

function endQuiz() {
    screens.game.classList.remove('active');
    screens.end.classList.add('active');
    
    const total = quizData[currentSet].questions.length;
    const percentage = Math.round((score / total) * 100);

    document.getElementById('final-score').textContent =
        `${score}/${total} (${percentage}%)`;
    
    let message = '';
    if (percentage === 100) {
        message = '👑 Xuất sắc! Bạn là thiên tài!';
    } else if (percentage >= 70) {
        message = '👍 Rất tốt! Giỏi lắm!';
    } else if (percentage >= 40) {
        message = '👌 Khá ổn! Cố lên nhé!';
    } else {
        message = '😅 Chơi lại để cải thiện nào!';
    }
    
    document.getElementById('final-title').textContent = percentage === 100 ? '🏆 CHÚC MỪNG!' : '🎉 Hoàn Thành!';
    document.getElementById('final-message').textContent = message;
}

document.getElementById('restart-btn').addEventListener('click', () => {
    screens.end.classList.remove('active');
    screens.menu.classList.add('active');
    currentSet = -1;
});