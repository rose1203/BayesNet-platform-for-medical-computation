/* modal.js */

$(document).ready(function(){

    // 显示/隐藏遮罩层的方法
    var showModal = function(){
        $('.modal_bg').fadeIn();
    };
    var hideModal = function(){
        $('.modal_bg').fadeOut();
    };

    // 点击按钮，弹出遮罩层
    $('#modal_btn').on('click', showModal);

    // 点击关闭按钮，关闭遮罩层
    $('#modal_close').on('click', hideModal);

});
