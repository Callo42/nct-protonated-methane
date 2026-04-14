      SUBROUTINE calcangle(a,b,c,theta)
!.... program to calculate theta
!.... a, b, c are the length of sides of triangle

      implicit none

      double precision :: a, b, c, theta, pi, d

      pi = 4.d0*datan(1.d0)

      if (abs(b-c-a).le.1e-5) then
         theta = 0.d0
      else
         d = b**2 + c**2 - a**2
         theta = 180.d0*dacos(d/(2.d0*b*c))/pi
      end if

      END SUBROUTINE calcangle
