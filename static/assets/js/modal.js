/* modal.js */

$(document).ready(function(){

    // ��ʾ/�������ֲ�ķ���
    var showModal = function(){
        $('.modal_bg').fadeIn();
    };
    var hideModal = function(){
        $('.modal_bg').fadeOut();
    };

    // �����ť���������ֲ�
    $('#modal_btn').on('click', showModal);

    // ����رհ�ť���ر����ֲ�
    $('#modal_close').on('click', hideModal);

});
