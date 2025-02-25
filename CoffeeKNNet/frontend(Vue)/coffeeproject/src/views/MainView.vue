<template>
  <div class="main-view bg-image">
    <div class="center-content">
      <h4 class="wow fadeInUp" data-wow-duration="1s" data-wow-delay="0.5s">
        Check Grade Via
      </h4>
      <h1 class="wow fadeIn" data-wow-duration="1s" data-wow-delay="0.3s">
        <span class="coffee">Coffee</span>
        <span class="knn">KNN</span>
        <span class="et">et</span>
      </h1>
      <h3 class="wow fadeIn" data-wow-duration="1s" data-wow-delay="0.9s">커피 원두의 등급을 예측할 수 있어요.</h3>
      <a href="CoffeeFormView" class="chkbtn wow fadeInUp" data-wow-duration="1s" data-wow-delay="1.5s">
        CHECK NOW&nbsp;▶
      </a>
    </div>
  </div>
</template>
<style>
.main-view {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 9999;
}

.center-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
}

.coffee {
  color: black;
  font-weight: 900;
  font-size: 100px;
}

.knn {
  color: #60403b;
  padding: 2px 5px;
  border-radius: 3px;
  font-weight: 900;
  font-size: 100px;
}

.et {
  color: black;
  font-weight: 900;
  font-size: 100px;
}

.bg-image {
  background-image: linear-gradient(0deg,
      rgba(217, 217, 217, 0.55),
      rgba(217, 217, 217, 0.55)),
    url("~@/assets/coffeebeans.png");
  background-position: 50% 50%;
  background-size: cover;
  background-repeat: no-repeat;
}

.chkbtn {
  background-color: #60403b;
  color: white;
  margin: 20px 20px;
  padding: 15px;
  border-radius: 5px;
  font-weight: 700;
  font-size: 25px;
}

.chkbtn:hover {
  background-color: black;
  cursor: pointer;
  color: white;
}
</style>
