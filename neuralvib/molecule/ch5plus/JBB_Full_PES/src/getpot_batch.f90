! Free-form Fortran 90; no column rules headache.
! Uses your existing scalar FUNCTION getpot_with_return(xn) from the .f code.

subroutine getpot_batch(xb, bsz, epots)
  implicit none
  integer, parameter :: natoms = 6
  integer, intent(in) :: bsz
  double precision, intent(in)  :: xb(3, natoms, bsz)
  double precision, intent(out) :: epots(bsz)
  integer :: i
  double precision getpot_with_return
  external getpot_with_return

!$omp parallel do default(shared) private(i)
  do i = 1, bsz
    epots(i) = getpot_with_return(xb(:,:,i))
  end do
!$omp end parallel do
end subroutine getpot_batch
