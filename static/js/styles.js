$(function () {
    $("button").click(function () {
        $(".cover").fadeIn("300");
    })
    $(".cover,.close").click(function () {
        $(".cover").fadeOut("300");
    })
    $(".contents").click(function (e) {
        e.stopPropagation();
    })
})

function updateRangeInput(elem) {
    $(elem).next().val($(elem).val());
}