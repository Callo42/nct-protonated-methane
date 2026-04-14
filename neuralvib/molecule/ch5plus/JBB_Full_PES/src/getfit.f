! add another '!' to some comment lines because of conflict with omp
!***********************************************************************
      SUBROUTINE getfit(xn,f0)
!
      implicit none
!
      include "fit.inc"
!
      double precision, dimension(0:2,0:5) :: xn
      double precision, dimension(0:2,0:5) :: xn1
      double precision, dimension(0:2,0:5) :: gf0
      double precision, dimension(0:5,0:5) :: d0, r0
      double precision, dimension(0:2) :: rvec
      double precision, dimension(0:m+2*mr-1) :: vec,vec0,vec1
      double precision :: f0 
      double precision :: t0,t1
      double precision, parameter :: dd = 1.0d-6
!
      integer :: i, j, l, k
!
!     -----------------------------------------------------------------
! The following lines are the code that evaluates my fitted function.
! The xnuc are the nuclear coordinates, as found in your table.
! Routine getd0 evaluates my internal coordinates d0(0:5,0:5), which are
! functions of the interparticle distances.  I've defined d0 both above
! and below the diagonal, but it is symmetric (d0(i,j)=d0(j,i)), and d0
! vanishes on the diagonal.  Then routine vec evaluates the (m) basis
! functions at the configuration described by d0, and finally f0 is the
! fitted potential.  The basis functions are polynomials of the entries
! in d0, obeying all the expected symmetries.
!

      call getd0 (xn, dc0, dc1, dw0, dw1, d0, r0)
      call getvec (m, d0, vec(0:m-1))
      call getrvec (3, r0, rvec)
      do l = 0, mr-1
         do k = 0, 1
            vec(m+2*l+k) = rvec(k+1)*vec(l)
         enddo
      enddo

      f0 = dot_product(coef,vec)

!.... Now the code is serving for PE calculation, gradient
!...  is ignored for decreasing computational cost

!       do i = 0, 2
!          do j = 0, 5
!             xn1 = xn ; xn1(i,j) = xn1(i,j)-dd
!             call getd0 (xn1, dc0, dc1, dw0, dw1, d0, r0)
!             call getvec (m, d0, vec(0:m-1))
!             call getrvec (3, r0, rvec)
!             do l = 0, mr-1
!                do k = 0, 1
!                vec(m+2*l+k) = rvec(k+1)*vec(l)
!             enddo
!          enddo
!           t0 = dot_product(coef,vec)

!           xn1 = xn ; xn1(i,j) = xn1(i,j)+dd
!           call getd0 (xn1, dc0, dc1, dw0, dw1, d0, r0)
!           call getvec (m, d0, vec(0:m-1))
!           call getrvec (3, r0, rvec)
!          do l = 0, mr-1
!             do k = 0, 1
!                vec(m+2*l+k) = rvec(k+1)*vec(l)
!             enddo
!          enddo
!           t1 = dot_product(coef,vec)

!
! Note that gf0 will be the negative gradient
!
!           gf0(i,j) = -(t1-t0)/(2*dd)
!         enddo
!       enddo
!
      END SUBROUTINE getfit

!********************************************************************
      SUBROUTINE getd0 (xn, dc0, dc1, dw0, dw1, d0, r0)

      implicit none

      double precision, dimension(0:2,0:5) :: xn
      double precision :: dc0, dc1, dw0, dw1
      double precision, dimension(0:5,0:5) :: d0, r0
      double precision, parameter :: p = 2.0d0
      integer :: i, j
      double precision :: t0

! Note: CH5+.  Nuclei 0..5; i=0 for the C, i in 1..5 for the H.

      d0(0,0) = 0.d0
      r0(0,0) = 0.d0

      do j = 1, 5
         t0 = sqrt((xn(0,j)-xn(0,0))**2+(xn(1,j)-xn(1,0))**2+ 
     $     (xn(2,j)-xn(2,0))**2)
!         d0(0,j) = (log(t0)-dc0)/dw0
!         d0(0,j) = (dexp(-t0/3.d0)-dc0)/dw0
         d0(0,j) = (dexp(-t0/p)-dc0)/dw0
         d0(j,0) = d0(0,j)
         r0(0,j) = t0
         r0(j,0) = t0
      end do

      do i = 1, 5
         d0(i,i) = 0.d0
         r0(i,i) = 0.d0
         do j = i+1, 5
            t0 = sqrt((xn(0,j)-xn(0,i))**2+(xn(1,j)-xn(1,i))**2+ 
     $        (xn(2,j)-xn(2,i))**2)
!           d0(i,j) = (log(t0)-dc1)/dw1
!           d0(i,j) = (dexp(-t0/p)-dc1)/dw1
           d0(i,j) = (dexp(-t0/p)-dc1)/dw1
            d0(j,i) = d0(i,j)
            r0(i,j) = t0
            r0(j,i) = t0
         end do
      end do
      END SUBROUTINE getd0

!**********************************************************************
      SUBROUTINE getvec (m, d, vec)

      implicit none

      integer :: m

      double precision, dimension(0:5,0:5) :: d
      double precision, dimension(0:m-1) :: vec

      integer :: j0, j1, j2, j3, j4, j5, j, k, k0
      double precision,dimension(0:1) :: x
      double precision,dimension(0:3) :: y
      double precision,dimension(0:9) :: z
      double precision,dimension(0:21) :: u
      double precision,dimension(0:41) :: v
      double precision,dimension(0:59) :: w
      double precision,dimension(0:5,0:5) :: d2,d3,d4,d5
      double precision :: t0
      double precision :: her2, her3, her4, her5, her6, her7

      her2(t0) = (4*t0**2-2)/sqrt(dble(8*2))
      her3(t0) = (8*t0**2-12)*t0/sqrt(dble(16*6))
      her4(t0) = ((16*t0**2-48)*t0**2+12)/sqrt(dble(32*24))
      her5(t0) = ((32*t0**2-160)*t0**2+120)*t0/sqrt(dble(64*120))
      her6(t0) = (((64*t0**2-480)*t0**2+720)*t0**2-120)/sqrt(dble(
     $    128*720))
      her7(t0) = (((128*t0**2-1344)*t0**2+3360)*t0**2-1680)*t0/ 
     $    sqrt(dble(256*5040))
!     ------------------------------------------------------------------
!     Test for compatibility
      if (.not.(m.eq.1.or.m.eq.3.or.m.eq.10.or.m.eq.32.or. 
     $   m.eq.101.or.m.eq.299.or.m.eq.849.or.m.eq.2239)) then
       stop 'getvec - wrong dimension'
      endif

!     Computation
      x = 0 ; y = 0 ; z = 0 ; u = 0 ; v = 0 ; w = 0
      do j0 = 0, 5
       do j1 = j0+1, 5
        d2(j0,j1) = her2(d(j0,j1))
        d2(j1,j0) = d2(j0,j1)
        d3(j0,j1) = her3(d(j0,j1))
        d3(j1,j0) = d3(j0,j1)
        d4(j0,j1) = her4(d(j0,j1))
        d4(j1,j0) = d4(j0,j1)
        d5(j0,j1) = her5(d(j0,j1))
        d5(j1,j0) = d5(j0,j1)
       enddo
      enddo

      do j1 = 1, 5
       x(0) = x(0)+d(0,j1)/5
       y(0) = y(0)+d2(0,j1)/5
       z(0) = z(0)+d3(0,j1)/5
       u(0) = u(0)+d4(0,j1)/5
       v(0) = v(0)+d5(0,j1)/5
!!$       w() = w()+her6(d(0,j1))/5
       do j2 = 1, 5
       if (j2.ne.j1) then
        y(1) = y(1)+d(0,j1)*d(j1,j2)/20
        z(1) = z(1)+d2(0,j1)*d(j1,j2)/20
        z(2) = z(2)+d(0,j1)*d2(j1,j2)/20
        z(3) = z(3)+d(0,j1)*d(0,j2)*d(j1,j2)/20
        u(1) = u(1)+d3(0,j1)*d(j1,j2)/20
        u(2) = u(2)+d2(0,j1)*d2(j1,j2)/20
        u(3) = u(3)+d(0,j1)*d3(j1,j2)/20
        u(4) = u(4)+d2(0,j1)*d(0,j2)*d(j1,j2)/20
        u(5) = u(5)+d(0,j1)*d(0,j2)*d2(j1,j2)/20
        v(1) = v(1)+d4(0,j1)*d(j1,j2)/20
        v(2) = v(2)+d3(0,j1)*d2(j1,j2)/20
        v(3) = v(3)+d2(0,j1)*d3(j1,j2)/20
        v(4) = v(4)+d(0,j1)*d4(j1,j2)/20
        v(5) = v(5)+d3(0,j1)*d(0,j2)*d(j1,j2)/20
        v(6) = v(6)+d2(0,j1)*d(0,j2)*d2(j1,j2)/20
        v(7) = v(7)+d(0,j1)*d(0,j2)*d3(j1,j2)/20
        w(0) = w(0)+d4(0,j1)*d2(j1,j2)/20
        w(1) = w(1)+d3(0,j1)*d3(j1,j2)/20
        w(2) = w(2)+d2(0,j1)*d4(j1,j2)/20
        w(3) = w(3)+d(0,j1)*d5(j1,j2)/20
        w(4) = w(4)+d3(0,j1)*d2(0,j2)*d(j1,j2)/20
        w(5) = w(5)+d3(0,j1)*d(0,j2)*d2(j1,j2)/20
        w(6) = w(6)+d2(0,j1)*d(0,j2)*d3(j1,j2)/20
        w(7) = w(7)+d(0,j1)*d(0,j2)*d4(j1,j2)/20
!!$        w() = w()+d5(0,j1)*d(j1,j2)/20
!!$        w() = w()+d4(0,j1)*d(0,j2)*d(j1,j2)/20
!!$        w() = w()+d2(0,j1)*d2(0,j2)*d2(j1,j2)/20
        do j3 = 1, 5
        if (j3.ne.j2.and.j3.ne.j1) then
         z(4) = z(4)+d(0,j1)*d(j1,j2)*d(j2,j3)/60
         z(5) = z(5)+d(0,j1)*d(j1,j2)*d(j1,j3)/60
         u(6) = u(6)+d(0,j1)*d2(j1,j2)*d(j2,j3)/60
         u(7) = u(7)+d(0,j1)*d(j1,j2)*d2(j2,j3)/60
         u(8) = u(8)+d(0,j1)*d2(j1,j2)*d(j1,j3)/60
         u(9) = u(9)+d(0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)/60
         u(10) = u(10)+d2(0,j1)*d(j1,j2)*d(j2,j3)/60
         u(11) = u(11)+d2(0,j1)*d(j1,j2)*d(j1,j3)/60
         u(12) = u(12)+d(0,j1)*d(0,j2)*d(j1,j3)*d(j2,j3)/60
         v(8) = v(8)+d3(0,j1)*d(j1,j2)*d(j2,j3)/60
         v(9) = v(9)+d2(0,j1)*d2(j1,j2)*d(j2,j3)/60
         v(10) = v(10)+d2(0,j1)*d(j1,j2)*d2(j2,j3)/60
         v(11) = v(11)+d(0,j1)*d3(j1,j2)*d(j2,j3)/60
         v(12) = v(12)+d(0,j1)*d2(j1,j2)*d2(j2,j3)/60
         v(13) = v(13)+d(0,j1)*d(j1,j2)*d3(j2,j3)/60
         v(14) = v(14)+d3(0,j1)*d(j1,j2)*d(j1,j3)/60
         v(15) = v(15)+d2(0,j1)*d2(j1,j2)*d(j1,j3)/60
         v(16) = v(16)+d(0,j1)*d3(j1,j2)*d(j1,j3)/60
         v(17) = v(17)+d(0,j1)*d2(j1,j2)*d2(j1,j3)/60
         v(18) = v(18)+d2(0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)/60
         v(19) = v(19)+d(0,j1)*d2(j1,j2)*d(j1,j3)*d(j2,j3)/60
         v(20) = v(20)+d(0,j1)*d(j1,j2)*d(j1,j3)*d2(j2,j3)/60
         v(21) = v(21)+d2(0,j1)*d(0,j2)*d(j1,j3)*d(j2,j3)/60
         v(22) = v(22)+d(0,j1)*d(0,j2)*d2(j1,j3)*d(j2,j3)/60
         v(23) = v(23)+d(0,j1)*d(0,j2)*d(j1,j3)*d(j2,j3)*d(j1,j2)/60
         v(24) = v(24)+d(0,j1)*d(0,j2)*d(j1,j2)*d2(j1,j3)/60
!!$         w() = w()+d4(0,j1)*d(j1,j2)*d(j2,j3)/60
         w(8) = w(8)+d(0,j1)*d(j1,j2)*d4(j2,j3)/60
         w(9) = w(9)+d(0,j1)*d4(j1,j2)*d(j2,j3)/60
         w(10) = w(10)+d3(0,j1)*d2(j1,j2)*d(j2,j3)/60
         w(11) = w(11)+d(0,j1)*d2(j1,j2)*d3(j2,j3)/60
         w(12) = w(12)+d2(0,j1)*d3(j1,j2)*d(j2,j3)/60
         w(13) = w(13)+d(0,j1)*d3(j1,j2)*d2(j2,j3)/60
         w(14) = w(14)+d3(0,j1)*d(j1,j2)*d2(j2,j3)/60
         w(15) = w(15)+d2(0,j1)*d(j1,j2)*d3(j2,j3)/60
         w(16) = w(16)+d2(0,j1)*d2(j1,j2)*d2(j2,j3)/60
!!$         w() = w()+d4(0,j1)*d(j1,j2)*d(j1,j3)/60
         w(17) = w(17)+d(0,j1)*d4(j1,j2)*d(j1,j3)/60
         w(18) = w(18)+d3(0,j1)*d2(j1,j2)*d(j1,j3)/60
         w(19) = w(19)+d2(0,j1)*d3(j1,j2)*d(j1,j3)/60
         w(20) = w(20)+d(0,j1)*d3(j1,j2)*d2(j1,j3)/60
         w(21) = w(21)+d2(0,j1)*d2(j1,j2)*d2(j1,j3)/60
         w(22) = w(22)+d3(0,j1)*d(0,j2)*d(j1,j3)*d(j2,j3)/60
         w(23) = w(23)+d(0,j1)*d(0,j2)*d3(j1,j3)*d(j2,j3)/60
         w(44) = w(44)+d2(0,j1)*d2(0,j2)*d(j1,j3)*d(j2,j3)/60
         w(24) = w(24)+d2(0,j1)*d(0,j2)*d2(j1,j3)*d(j2,j3)/60
         w(25) = w(25)+d(0,j1)*d(0,j2)*d2(j1,j3)*d2(j2,j3)/60
         w(26) = w(26)+d2(0,j1)*d(0,j2)*d(j1,j3)*d2(j2,j3)/60
         w(45) = w(45)+d3(0,j1)*d(0,j2)*d(j1,j2)*d(j2,j3)/60
         w(27) = w(27)+d(0,j1)*d(j1,j2)*d(j1,j3)*d3(j2,j3)/60
         w(28) = w(28)+d(0,j1)*d(0,j2)*d3(j1,j2)*d(j2,j3)/60
         w(46) = w(46)+d3(0,j1)*d(0,j2)*d(j1,j2)*d(j1,j3)/60
!!$         w() = w()+d3(0,j1)*d(0,j2)*d(0,j3)*d(j1,j2)/60
         w(29) = w(29)+d(0,j1)*d3(j1,j2)*d(j1,j3)*d(j2,j3)/60
!!$         w() = w()+d(0,j1)*d(0,j2)*d(j1,j2)*d3(j2,j3)/60
         w(30) = w(30)+d3(0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)/60
         w(31) = w(31)+d2(0,j1)*d(0,j2)*d2(j1,j2)*d(j2,j3)/60
!!$         w() = w()+d2(0,j1)*d2(0,j2)*d(j1,j2)*d(j2,j3)/60
!!$         w() = w()+d2(0,j1)*d(0,j2)*d(0,j3)*d2(j1,j2)/60
         w(32) = w(32)+d(0,j1)*d2(j1,j2)*d(j1,j3)*d2(j2,j3)/60
!!$         w() = w()+d2(0,j1)*d(0,j2)*d(j1,j2)*d2(j2,j3)/60
         w(33) = w(33)+d2(0,j1)*d(j1,j2)*d(j1,j3)*d2(j2,j3)/60
!!$         w() = w()+d2(0,j1)*d(0,j2)*d2(j1,j2)*d(j1,j3)/60
!!$         w() = w()+d2(0,j1)*d2(0,j2)*d(0,j3)*d(j1,j2)/60
         w(34) = w(34)+d(0,j1)*d2(j1,j2)*d2(j1,j3)*d(j2,j3)/60
!!$         w() = w()+d(0,j1)*d(0,j2)*d2(j1,j2)*d2(j2,j3)/60
!!$         w() = w()+d(0,j1)*d2(0,j2)*d(j1,j2)*d2(j2,j3)/60
         w(35) = w(35)+d2(0,j1)*d(j1,j2)*d2(j1,j3)*d(j2,j3)/60
         w(36) = w(36)+d2(0,j1)*d(0,j2)*d(0,j3)*d(j1,j2)*d(j2,j3)/60
!!$         w() = w()+d2(0,j1)*d(0,j2)*d(j1,j2)*d(j1,j3)*d(j2,j3)/60
!!$         w() = w()+d(0,j1)*d(0,j2)*d(0,j3)*d2(j1,j2)*d(j2,j3)/60
         w(37) = w(37)+d(0,j1)*d(0,j2)*d(j1,j2)*d2(j1,j3)*d(j2,j3)/60
         w(38) = w(38)+d2(0,j1)*d(0,j2)*d(0,j3)*d(j1,j2)*d(j1,j3)/60
         w(39) = w(39)+d(0,j1)*d(0,j2)*d2(j1,j2)*d(j1,j3)*d(j2,j3)/60
!!$       w() = w()+d(0,j1)*d(0,j2)*d(0,j3)*d(j1,j2)*d(j1,j3)*d(j2,j3)/60
         do j4 = 1, 5
         if (j4.ne.j3.and.j4.ne.j2.and.j4.ne.j1) then
          u(13) = u(13)+d(0,j1)*d(j1,j2)*d(j1,j3)*d(j3,j4)/120
          u(14) = u(14)+d(0,j1)*d(j1,j2)*d(j1,j3)*d(j1,j4)/120
          v(25) = v(25)+d(0,j1)*d2(j1,j2)*d(j2,j3)*d(j3,j4)/120
          v(26) = v(26)+d2(0,j1)*d(j1,j2)*d(j1,j3)*d(j3,j4)/120
          v(27) = v(27)+d(0,j1)*d2(j1,j2)*d(j1,j3)*d(j3,j4)/120
          v(28) = v(28)+d(0,j1)*d(j1,j2)*d2(j1,j3)*d(j3,j4)/120
          v(29) = v(29)+d(0,j1)*d(j1,j2)*d(j1,j3)*d2(j3,j4)/120
          v(30) = v(30)+d2(0,j1)*d(j1,j2)*d(j1,j3)*d(j1,j4)/120
          v(31) = v(31)+d(0,j1)*d2(j1,j2)*d(j1,j3)*d(j1,j4)/120
          w(40) = w(40)+d(0,j1)*d3(j1,j2)*d(j1,j3)*d(j1,j4)/120
          w(41) = w(41)+d2(0,j1)*d(j1,j2)*d2(j2,j3)*d(j3,j4)/120
          w(42) = w(42)+d(0,j1)*d(j1,j2)*d2(j2,j3)*d2(j3,j4)/120
          w(43) = w(43)+d(0,j1)*d2(j1,j2)*d2(j1,j3)*d(j1,j4)/120
          w(44) = w(44)+d(0,j1)*d(j1,j2)*d(j2,j3)*d3(j3,j4)/120
          w(45) = w(45)+d(0,j1)*d2(j1,j2)*d(j2,j3)*d2(j3,j4)/120
          w(46) = w(46)+d(0,j1)*d(j1,j2)*d(j1,j3)*d2(j2,j3)*d(j1,j4)/120
!!$          w() = w()+d3(0,j1)*d(j1,j2)*d(j2,j3)*d(j3,j4)/120
!!$          w() = w()+d2(0,j1)*d2(j1,j2)*d(j2,j3)*d(j3,j4)/120
!!$          w() = w()+d3(0,j1)*d(j1,j2)*d(j1,j3)*d(j1,j4)/120
!!$          w() = w()+d2(0,j1)*d2(j1,j2)*d(j1,j3)*d(j1,j4)/120
!!$          w() = w()+d(0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)*d2(j1,j4)/120
!!$          w() = w()+d2(0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)*d(j1,j4)/120
!!$          w() = w()+d(0,j1)*d2(j1,j2)*d(j1,j3)*d(j2,j3)*d(j1,j4)/120
         endif
         enddo
        endif
        enddo
       endif
       enddo
      enddo
      do j0 = 1, 5
       do j1 = 1, 5
       if (j1.ne.j0) then
        x(1) = x(1)+d(j0,j1)/4
        y(2) = y(2)+d2(j0,j1)/4
        z(6) = z(6)+d3(j0,j1)/4
        u(15) = u(15)+d4(j0,j1)/4
        v(32) = v(32)+d5(j0,j1)/4
!!$        w() = w()+her6(d(j0,j1))/4
        do j2 = 1, 5
        if (j2.ne.j1.and.j2.ne.j0) then
         y(3) = y(3)+d(j0,j1)*d(j1,j2)/12
         z(7) = z(7)+d2(j0,j1)*d(j1,j2)/12
         z(8) = z(8)+d(j0,j1)*d(j0,j2)*d(j1,j2)/12
         u(16) = u(16)+d3(j0,j1)*d(j1,j2)/12
         u(17) = u(17)+d2(j0,j1)*d2(j1,j2)/12
         u(18) = u(18)+d2(j0,j1)*d(j0,j2)*d(j1,j2)/12
         v(33) = v(33)+d4(j0,j1)*d(j1,j2)/12
         v(34) = v(34)+d3(j0,j1)*d2(j1,j2)/12
         v(35) = v(35)+d3(j0,j1)*d(j0,j2)*d(j1,j2)/12
         v(36) = v(36)+d2(j0,j1)*d2(j0,j2)*d(j1,j2)/12
         w(47) = w(47)+d4(j0,j1)*d2(j1,j2)/12
         w(48) = w(48)+d3(j0,j1)*d3(j1,j2)/12
         w(49) = w(49)+d2(j0,j1)*d2(j0,j2)*d2(j1,j2)/12
         w(50) = w(50)+d4(j0,j1)*d(j0,j2)*d(j1,j2)/12
!!$         w() = w()+d3(j0,j1)*d2(j0,j2)*d(j1,j2)/12
!!$         w() = w()+d5(j0,j1)*d(j1,j2)/12
         do j3 = 1, 5
         if (j3.ne.j2.and.j3.ne.j1.and.j3.ne.j0) then
          z(9) = z(9)+d(j0,j1)*d(j0,j2)*d(j0,j3)/24
          u(19) = u(19)+d2(j0,j1)*d(j0,j2)*d(j0,j3)/24
          u(20) = u(20)+d(j0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)/24
          u(21) = u(21)+d(j0,j1)*d(j0,j2)*d(j1,j3)*d(j2,j3)/24
          v(37) = v(37)+d3(j0,j1)*d(j0,j2)*d(j0,j3)/24
          v(38) = v(38)+d2(j0,j1)*d2(j0,j2)*d(j0,j3)/24
          v(39) = v(39)+d2(j0,j1)*d(j1,j2)*d(j1,j3)*d(j2,j3)/24
          v(40) = v(40)+d(j0,j1)*d2(j1,j2)*d(j1,j3)*d(j2,j3)/24
          v(41) = v(41)+d(j0,j1)*d(j1,j2)*d(j1,j3)*d2(j2,j3)/24
          w(51) = w(51)+d4(j0,j1)*d(j1,j2)*d(j2,j3)/24
          w(52) = w(52)+d(j0,j1)*d4(j1,j2)*d(j2,j3)/24
          w(53) = w(53)+d3(j0,j1)*d2(j1,j2)*d(j2,j3)/24
          w(54) = w(54)+d3(j0,j1)*d(j1,j2)*d2(j2,j3)/24
          w(55) = w(55)+d2(j0,j1)*d3(j1,j2)*d(j2,j3)/24
          w(56) = w(56)+d2(j0,j1)*d2(j1,j2)*d2(j2,j3)/24
          w(57) = w(57)+d4(j0,j1)*d(j0,j2)*d(j0,j3)/24
          w(58) = w(58)+d2(j0,j1)*d2(j0,j2)*d2(j0,j3)/24
          w(59) = w(59)+d3(j0,j1)*d2(j0,j2)*d(j0,j3)/24
         endif
         enddo
        endif
        enddo
       endif
       enddo
      enddo
      vec(0) = 1
      if (3.le.m) then
       vec(1) = x(0)
       vec(2) = x(1)
      endif
      if (10.le.m) then
       vec(3) = her2(x(0))
       vec(4) = x(0)*x(1)
       vec(5) = her2(x(1))
       do k = 0, 3
        vec(6+k) = y(k)
       enddo
      endif
!! third order terms
      if (32.le.m) then
       k0 = 10
       vec(k0) = her3(x(0))
       vec(k0+1) = her2(x(0))*x(1)
       vec(k0+2) = x(0)*her2(x(1))
       vec(k0+3) = x(0)*y(0)
       vec(k0+4) = x(0)*y(1)
       vec(k0+5) = x(0)*y(2)
       vec(k0+6) = x(0)*y(3)
       vec(k0+7) = her3(x(1))
       vec(k0+8) = x(1)*y(0)
       vec(k0+9) = x(1)*y(1)
       vec(k0+10) = x(1)*y(2)
       vec(k0+11) = x(1)*y(3)
       k0 = 22
       do k = 0, 9
        vec(k0+k) = z(k)
       enddo
      endif
!! fourth order terms
      if (101.le.m) then
       k0 = 32
       vec(k0) = her4(x(0))
       vec(k0+1) = her3(x(0))*x(1)
       vec(k0+2) = her2(x(0))*her2(x(1))
       do k = 0, 3
        vec(k0+3+k) = her2(x(0))*y(k)
       enddo
       vec(k0+7) = x(0)*her3(x(1))
       do k = 0, 3
        vec(k0+8+k) = x(0)*x(1)*y(k)
       enddo
       do k = 0, 9
        vec(k0+12+k) = x(0)*z(k)
       enddo
       vec(k0+22) = her4(x(1))
       do k = 0, 3
        vec(k0+23+k) = her2(x(1))*y(k)
       enddo
       do k = 0, 9
        vec(k0+27+k) = x(1)*z(k)
       enddo
       k0 = k0+37
       do j = 0, 3
        do k = j, 3
         vec(k0) = y(j)*y(k)
         k0 = k0+1
        enddo
       enddo
       if (k0.ne.79) then
        stop 'getvec: counting error'
       endif
       do k = 0, 21
        vec(k0+k) = u(k)
       enddo
      endif
!! fifth order terms
      if (299.le.m) then
       k0 = 101
       vec(k0) = her5(x(0))
       vec(k0+1) = her4(x(0))*x(1)
       vec(k0+2) = her3(x(0))*her2(x(1))
       do k = 0, 3
        vec(k0+3+k) = her3(x(0))*y(k)
       enddo
       vec(k0+7) = her2(x(0))*her3(x(1))
       do k = 0, 3
        vec(k0+8+k) = her2(x(0))*x(1)*y(k)
       enddo
       do k = 0, 9
        vec(k0+12+k) = her2(x(0))*z(k)
       enddo
       vec(k0+22) = x(0)*her4(x(1))
       do k = 0, 3
        vec(k0+23+k) = x(0)*her2(x(1))*y(k)
       enddo
       do k = 0, 9
        vec(k0+27+k) = x(0)*x(1)*z(k)
       enddo
       k0 = k0+37
       do j = 0, 3
        do k = j, 3
         vec(k0) = x(0)*y(j)*y(k)
         k0 = k0+1
        enddo
       enddo
       if (k0.ne.148) then
        stop 'getvec: counting error'
       endif
       do k = 0, 21
        vec(k0+k) = x(0)*u(k)
       enddo
       vec(k0+22) = her5(x(1))
       do k = 0, 3
        vec(k0+23+k) = her3(x(1))*y(k)
       enddo
       do k = 0, 9
        vec(k0+27+k) = her2(x(1))*z(k)
       enddo
       k0 = k0+37
       do j = 0, 3
        do k = j, 3
         vec(k0) = x(1)*y(j)*y(k)
         k0 = k0+1
        enddo
       enddo
       if (k0.ne.195) then
        stop 'getvec: counting error'
       endif
       do k = 0, 21
        vec(k0+k) = x(1)*u(k)
       enddo
       do j = 0, 3
        do k = 0, 9
         vec(k0+22+10*j+k) = y(j)*z(k)
        enddo
       enddo
       k0 = 257
       do k = 0, 41
        vec(k0+k) = v(k)
       enddo
      endif
!! sixth order terms
      if (849.le.m) then
       k0 = 299
       do k = 0, 197
        vec(k0+k) = x(0)*vec(101+k)
       enddo
       do k = 0, 128
        vec(k0+198+k) = x(1)*vec(170+k)
       enddo
       do k = 0, 31
        vec(k0+327+k) = y(0)*vec(69+k)
       enddo
       do k = 0, 27
        vec(k0+359+k) = y(1)*vec(73+k)
       enddo
       do k = 0, 24
        vec(k0+387+k) = y(2)*vec(76+k)
       enddo
       do k = 0, 22
        vec(k0+412+k) = y(3)*vec(78+k)
       enddo
       k0 = k0+435
       do j = 0, 9
        do k = j, 9
         vec(k0) = z(j)*z(k)
         k0 = k0+1
        enddo
       enddo
       if (k0.ne.789) then
        stop 'getvec: counting error'
       endif
       do k = 0, 59
        vec(k0+k) = w(k)
       enddo
      endif
!! seventh order terms (incomplete)
      if (2239.le.m) then
       k0 = 849
       do k = 0, 549 ! 849-299-1
        vec(k0+k) = x(0)*vec(299+k)
       enddo
       do k = 0, 351 ! 849-497-1
        vec(k0+550+k) = x(1)*vec(497+k)
       enddo
       do k = 0, 81 ! 299-217-1
        vec(k0+902+k) = y(0)*vec(217+k)
       enddo
       do k = 0, 71
        vec(k0+984+k) = y(1)*vec(227+k)
       enddo
       do k = 0, 61
        vec(k0+1056+k) = y(2)*vec(237+k)
       enddo
       do k = 0, 51
        vec(k0+1118+k) = y(3)*vec(247+k)
       enddo
       k0 = k0+1170
       do j = 0, 9
        do k = 0, 21
         vec(k0+22*j+k) = z(j)*u(k)
        enddo
       enddo
      endif
      return
      END SUBROUTINE getvec
!********************************************************
      SUBROUTINE getrvec (m, r, vec)

      implicit none

      integer m

      double precision, dimension(0:5,0:5) :: r, r1
      double precision, dimension(0:m-1) :: vec
      integer :: j0, j1, j2, i, j, k
      double precision, dimension(0:1) :: x
      double precision, dimension(0:3) :: y
!     ------------------------------------------------------------------
!     Test for compatibility
      if (.not.(m.eq.1.or.m.eq.3.or.m.eq.10)) then
       stop 'getrvec - wrong dimension'
      endif
!     Computation
      x = 0 ; y = 0
      do i = 0, 5
       do j = 0, 5
        if (i.eq.j) then
         r1(i,j) = 0
        else
         r1(i,j) = exp(-r(i,j))/r(i,j)
        endif
       enddo
      enddo
      do j1 = 1, 5
       x(0) = x(0)+r1(0,j1)/5
       y(0) = y(0)+r1(0,j1)**2/5
       do j2 = 1, 5
       if (j2.ne.j1) then
        y(1) = y(1)+r1(0,j1)*r1(j1,j2)/20
       endif
       enddo
      enddo
      do j0 = 1, 5
       do j1 = 1, 5
       if (j1.ne.j0) then
        x(1) = x(1)+r1(j0,j1)/4
        y(2) = y(2)+r1(j0,j1)**2/4
        do j2 = 1, 5
        if (j2.ne.j1.and.j2.ne.j0) then
         y(3) = y(3)+r1(j0,j1)*r1(j1,j2)/12
        endif
        enddo
       endif
       enddo
      enddo
      vec(0) = 1
      if (3.le.m) then
       vec(1) = x(0)
       vec(2) = x(1)
      endif
      if (10.le.m) then
       vec(3) = x(0)**2
       vec(4) = x(0)*x(1)
       vec(5) = x(1)**2
       do k = 0, 3
        vec(6+k) = y(k)
       enddo
      endif
      END SUBROUTINE getrvec
