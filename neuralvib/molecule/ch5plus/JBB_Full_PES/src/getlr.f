******************************************************************************
!
!.... Subroutine to obtain the long-range interaction
!
!.... 02/14/05   Zhong      SUBROUTINE created
!.... 02/22/05   Zhong      Modified for the long range effect 

      SUBROUTINE getlr(r,rh2,theta,elr)

      implicit none

      double precision :: r, rh2, theta
      double precision :: alpha, alpha0, alpha90
      double precision :: u3, u4, p2, q
      double precision :: epot, elr
      double precision :: pi

      pi = 4.d0 * datan(1.d0)
      call cspline(rh2,alpha0,alpha90,q) 

!      write(*,'a4,1x,f9.5')'rh2=',rh2
      theta = theta / 180.d0 * pi
      alpha = 1.d0/3.d0*(alpha0 + 2*alpha90)
      p2 = 0.5d0*(3*(cos(theta))**2 - 1.d0)
      u3 = q*p2/r**3
      u4 = -(0.5d0*alpha + 1.d0/3.d0*(alpha0 - alpha90)*p2)/r**4
      elr = u3 + u4 

      END SUBROUTINE getlr
