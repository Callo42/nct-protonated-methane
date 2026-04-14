*****************************************************************************
!...  subroutine for cubic spline
!...  driver for routine splint, which calls spline
!...  x ....       H-H bond distance
!     y0 ...       alpha parallel
!     y90 ...      alpha perpendicular
!     yq ...       quadrupole      
*****************************************************************************
      SUBROUTINE cspline(x,y0,y90,yq)

      implicit none

      integer, parameter :: NP = 21, NP2 = 257
      integer :: i, nfunc
      double precision :: x, y0, yp01, yp0n, y90, yp901, yp90n
      double precision :: ypq1, ypqn, yq
      double precision, dimension(NP) :: xa, ya0, ya90, y2s0, y2s90
      double precision, dimension(NP2) :: xa1, ya1, yq2

      open(unit=70,file='archive/support/alpha_1967.dat', status='old')
      open(unit=71,file='archive/support/q.dat', status='old')

      read(70,*)
      do i = 1, NP
         read(70,*) xa(i), ya0(i), ya90(i)
      end do
         yp01 = 0.d0
         yp0n = 0.d0
         yp901 = 0.d0
         yp90n = 0.d0

C     call SPLINE to get second derivatives
         call spline(xa,ya0,NP,yp01,yp0n,y2s0)
         call spline(xa,ya90,NP,yp901,yp90n,y2s90)

C     call SPLINT for interpolations
         call splint(xa,ya0,y2s0,NP,x,y0)
         call splint(xa,ya90,y2s90,NP,x,y90)

      do i = 1, NP2
         read(71,*) xa1(i), ya1(i)
      end do
         ypq1 = 0.d0
         ypqn = 0.d0

         call spline(xa1,ya1,NP2,ypq1,ypqn,yq2)
         call splint(xa1,ya1,yq2,NP2,x,yq)

      close(70)
      close(71)         
      END SUBROUTINE cspline
***********************************************************************
!...  Cubic spline code
!...  Original cubic spline code from Numerial Recipe has been modified
!...  to adapt double precision
!...  02/23/05    Zhong
!
***********************************************************************
      SUBROUTINE spline(x,y,n,yp1,ypn,y2)

      integer, parameter :: NMAX=500
      integer :: i, k

      double precision :: yp1,ypn
      double precision :: p, qn, sig, un
      double precision, dimension(n) :: x, y, y2
      double precision, dimension(NMAX) :: u
 
      if (yp1.gt..99e30) then
         y2(1)=0.5d0
         u(1)=0.5d0
      else
         y2(1)=-.5d0
         u(1)=(3./(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif

      do i = 2, n-1
         sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
         p=sig*y2(i-1)+2.d0
         y2(i)=(sig-1.)/p
         u(i)=(6.*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/
     $ (x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      end do

      if (ypn.gt..99e30) then
         qn=0.d0
         un=0.d0
      else
         qn=.5d0
         un=(3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif

      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.d0)

      do k = n-1, 1, -1
         y2(k)=y2(k)*y2(k+1)+u(k)
      end do
      END SUBROUTINE spline
****************************************************************************
      SUBROUTINE splint(xa,ya,y2a,n,x,y)
      integer :: n
      double precision :: x, y
      double precision, dimension(n) :: xa, y2a, ya

      integer :: k, khi, klo
      double precision :: a, b, h

      klo = 1
      khi = n

1     if (khi-klo.gt.1) then
         k=(khi+klo)/2
         if (xa(k).gt.x)then
            khi=k
         else
            klo=k
         endif
        goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.) pause 'bad xa input in splint'
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**
     $2)/6.d0
      END SUBROUTINE splint
